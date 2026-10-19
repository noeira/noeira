"""Chess — CPU+GPU environment for two-player self-play RL training.

Full chess rules: all piece moves, castling, en passant, pawn promotion,
check, checkmate, stalemate, 50-move draw rule.

State layout (STATE_SIZE = 72):
  [0..63]  board cells (piece IDs, row-major: sq = row*8 + col)
           Row 0 = rank 1 (white), Row 7 = rank 8 (black)
           0=empty, 1=wP, 2=wN, 3=wB, 4=wR, 5=wQ, 6=wK
                    7=bP, 8=bN, 9=bB, 10=bR, 11=bQ, 12=bK
  [64]     current_player (0=white, 1=black)
  [65]     castling_rights (4-bit: 1=wK, 2=wQ, 4=bK, 8=bQ)
  [66]     en_passant_sq (-1=none, else target square)
  [67]     halfmove_clock (50-move rule)
  [68]     fullmove_number
  [69]     game_result (0=ongoing, 1=white wins, 2=black wins, 3=draw)
  [70]     white_king_sq
  [71]     black_king_sq

Canonical obs (OBS_DIM = 896 = 14 planes × 64):
  Planes 0-5:   my pieces by type (pawn, knight, bishop, rook, queen, king)
  Planes 6-11:  opponent pieces by type
  Plane 12:     my castling rights (all 1s if kingside, all 1s if queenside)
  Plane 13:     en passant plane (1.0 at en passant target square)

Actions: AlphaZero encoding = 4672 (64 squares × 73 move types).
  Types 0-55:  queen-like moves (8 directions × 7 distances)
  Types 56-63: knight moves (8 offsets)
  Types 64-72: underpromotions (3 directions × 3 pieces)
  Queen promotions use the queen-move encoding.
  Actions are in CANONICAL coordinates (flipped for black).
"""

from std.random import random_float64
from noeira.nn.core.ptr import untracked
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import dealloc, unsafe_memset, alloc
from std.ffi import c_int, c_float
from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    TwoPlayerDiscreteEnv,
    GPUTwoPlayerDiscreteEnv,
    RenderableEnv,
)
from .chess_sprites import (
    create_sprite_sheet,
    PIECE_SIZE as SPRITE_SIZE,
    SHEET_WIDTH as SPRITE_SHEET_WIDTH,
    SHEET_HEIGHT as SPRITE_SHEET_HEIGHT,
    BYTES_PER_PIXEL as SPRITE_BPP,
)
from noeira.render import Renderer2D, SDL_Color
from ..core.board_env import BoardGameState, BoardGameAction, board_dtype
from noeira.core.tiles import Tile, ImmTile, tile

# ============================================================================
# Piece constants
# ============================================================================
comptime EMPTY: Int = 0
comptime W_PAWN: Int = 1
comptime W_KNIGHT: Int = 2
comptime W_BISHOP: Int = 3
comptime W_ROOK: Int = 4
comptime W_QUEEN: Int = 5
comptime W_KING: Int = 6
comptime B_PAWN: Int = 7
comptime B_KNIGHT: Int = 8
comptime B_BISHOP: Int = 9
comptime B_ROOK: Int = 10
comptime B_QUEEN: Int = 11
comptime B_KING: Int = 12

# State indices
comptime S_PLAYER: Int = 64
comptime S_CASTLING: Int = 65
comptime S_EP: Int = 66
comptime S_HALFMOVE: Int = 67
comptime S_FULLMOVE: Int = 68
comptime S_RESULT: Int = 69
comptime S_WK: Int = 70
comptime S_BK: Int = 71

# Castling bits
comptime CASTLE_WK: Int = 1
comptime CASTLE_WQ: Int = 2
comptime CASTLE_BK: Int = 4
comptime CASTLE_BQ: Int = 8

# Game results
comptime RESULT_ONGOING: Int = 0
comptime RESULT_WHITE_WINS: Int = 1
comptime RESULT_BLACK_WINS: Int = 2
comptime RESULT_DRAW: Int = 3

# ============================================================================
# Direction tables
# ============================================================================


# Queen-like directions: (dr, dc) — N, NE, E, SE, S, SW, W, NW
def _queen_dr(d: Int) -> Int:
    if d == 0:
        return 1  # N
    if d == 1:
        return 1  # NE
    if d == 2:
        return 0  # E
    if d == 3:
        return -1  # SE
    if d == 4:
        return -1  # S
    if d == 5:
        return -1  # SW
    if d == 6:
        return 0  # W
    return 1  # NW


def _queen_dc(d: Int) -> Int:
    if d == 0:
        return 0  # N
    if d == 1:
        return 1  # NE
    if d == 2:
        return 1  # E
    if d == 3:
        return 1  # SE
    if d == 4:
        return 0  # S
    if d == 5:
        return -1  # SW
    if d == 6:
        return -1  # W
    return -1  # NW


# Knight offsets: (dr, dc)
def _knight_dr(k: Int) -> Int:
    if k == 0:
        return 2
    if k == 1:
        return 1
    if k == 2:
        return -1
    if k == 3:
        return -2
    if k == 4:
        return -2
    if k == 5:
        return -1
    if k == 6:
        return 1
    return 2


def _knight_dc(k: Int) -> Int:
    if k == 0:
        return 1
    if k == 1:
        return 2
    if k == 2:
        return 2
    if k == 3:
        return 1
    if k == 4:
        return -1
    if k == 5:
        return -2
    if k == 6:
        return -2
    return -1


# ============================================================================
# Helpers
# ============================================================================


@always_inline
def _on_board(r: Int, c: Int) -> Bool:
    return r >= 0 and r < 8 and c >= 0 and c < 8


@always_inline
def _sq(r: Int, c: Int) -> Int:
    return r * 8 + c


@always_inline
def _row(sq: Int) -> Int:
    return sq // 8


@always_inline
def _col(sq: Int) -> Int:
    return sq % 8


@always_inline
def _flip_sq(sq: Int) -> Int:
    """Flip square vertically (for canonical view)."""
    return (7 - _row(sq)) * 8 + _col(sq)


@always_inline
def _is_white(piece: Int) -> Bool:
    return piece >= 1 and piece <= 6


@always_inline
def _is_black(piece: Int) -> Bool:
    return piece >= 7 and piece <= 12


@always_inline
def _piece_type(piece: Int) -> Int:
    """Return piece type 1-6 regardless of color."""
    if piece <= 6:
        return piece
    return piece - 6


@always_inline
def _is_friendly(piece: Int, player: Int) -> Bool:
    if player == 0:
        return _is_white(piece)
    return _is_black(piece)


@always_inline
def _is_enemy(piece: Int, player: Int) -> Bool:
    if piece == EMPTY:
        return False
    return not _is_friendly(piece, player)


@always_inline
def _make_piece(piece_type: Int, player: Int) -> Int:
    """Create piece of given type (1-6) for player."""
    if player == 0:
        return piece_type
    return piece_type + 6


# ============================================================================
# Move struct (internal representation)
# ============================================================================


struct Move(Copyable, ImplicitlyCopyable, Movable):
    var from_sq: Int
    var to_sq: Int
    var promo: Int  # 0=none, 2=knight, 3=bishop, 4=rook, 5=queen

    def __init__(out self, from_sq: Int, to_sq: Int, promo: Int = 0):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.promo = promo

    def __init__(out self, *, copy: Self):
        self.from_sq = copy.from_sq
        self.to_sq = copy.to_sq
        self.promo = copy.promo

    def __init__(out self, *, deinit move: Self):
        self.from_sq = move.from_sq
        self.to_sq = move.to_sq
        self.promo = move.promo


# ============================================================================
# Action encoding / decoding (AlphaZero 4672)
# ============================================================================


def _encode_action(m: Move, player: Int) -> Int:
    """Encode a move in real coordinates to a canonical action index."""
    var from_sq = m.from_sq
    var to_sq = m.to_sq
    if player == 1:
        from_sq = _flip_sq(from_sq)
        to_sq = _flip_sq(to_sq)

    var fr = _row(from_sq)
    var fc = _col(from_sq)
    var tr = _row(to_sq)
    var tc = _col(to_sq)
    var dr = tr - fr
    var dc = tc - fc

    # Underpromotion?
    if m.promo != 0 and m.promo != 5:  # not queen promo, not none
        var dir_idx: Int
        if dc == -1:
            dir_idx = 0  # left capture
        elif dc == 0:
            dir_idx = 1  # straight
        else:
            dir_idx = 2  # right capture
        var piece_idx: Int
        if m.promo == 2:
            piece_idx = 0  # knight
        elif m.promo == 3:
            piece_idx = 1  # bishop
        else:
            piece_idx = 2  # rook
        var move_type = 64 + dir_idx * 3 + piece_idx
        return from_sq * 73 + move_type

    # Knight move?
    for k in range(8):
        if _knight_dr(k) == dr and _knight_dc(k) == dc:
            return from_sq * 73 + 56 + k

    # Queen-like move (including pawn moves, king moves, queen promos)
    # Must be along a straight line or diagonal
    var abs_dr = dr if dr >= 0 else -dr
    var abs_dc = dc if dc >= 0 else -dc

    # Validate: straight (one axis is 0) or diagonal (both axes equal)
    if abs_dr != 0 and abs_dc != 0 and abs_dr != abs_dc:
        return -1  # not a valid queen-like move

    var dist = max(abs_dr, abs_dc)
    if dist == 0 or dist > 7:
        return -1

    var norm_dr = 0
    if dr > 0:
        norm_dr = 1
    elif dr < 0:
        norm_dr = -1
    var norm_dc = 0
    if dc > 0:
        norm_dc = 1
    elif dc < 0:
        norm_dc = -1

    var dir_idx = -1
    for d in range(8):
        if _queen_dr(d) == norm_dr and _queen_dc(d) == norm_dc:
            dir_idx = d
            break

    if dir_idx < 0:
        return -1

    var move_type = dir_idx * 7 + (dist - 1)
    return from_sq * 73 + move_type


def _decode_action(action: Int, player: Int) -> Move:
    """Decode a canonical action index to a move in real coordinates."""
    var canonical_from = action // 73
    var move_type = action % 73

    var fr = _row(canonical_from)
    var fc = _col(canonical_from)
    var tr: Int
    var tc: Int
    var promo = 0

    if move_type < 56:
        # Queen-like move
        var dir_idx = move_type // 7
        var dist = move_type % 7 + 1
        tr = fr + _queen_dr(dir_idx) * dist
        tc = fc + _queen_dc(dir_idx) * dist
        # Check for queen promotion (pawn reaching last rank)
        if tr == 7:
            promo = 5  # will be validated by caller
    elif move_type < 64:
        # Knight move
        var k = move_type - 56
        tr = fr + _knight_dr(k)
        tc = fc + _knight_dc(k)
    else:
        # Underpromotion
        var idx = move_type - 64
        var dir = idx // 3
        var piece = idx % 3
        tr = fr + 1  # always forward in canonical
        tc = fc + (dir - 1)  # -1, 0, +1
        if piece == 0:
            promo = 2  # knight
        elif piece == 1:
            promo = 3  # bishop
        else:
            promo = 4  # rook

    var from_sq = canonical_from
    var to_sq: Int
    if _on_board(tr, tc):
        to_sq = _sq(tr, tc)
    else:
        to_sq = -1  # invalid

    # Un-flip for black
    if player == 1:
        from_sq = _flip_sq(from_sq)
        if to_sq >= 0:
            to_sq = _flip_sq(to_sq)

    return Move(from_sq, to_sq, promo)


def _piece_char(piece: Int) -> String:
    """Return the character for a given piece ID."""
    if piece == W_KING:
        return "K"
    if piece == W_QUEEN:
        return "Q"
    if piece == W_ROOK:
        return "R"
    if piece == W_BISHOP:
        return "B"
    if piece == W_KNIGHT:
        return "N"
    if piece == W_PAWN:
        return "P"
    if piece == B_KING:
        return "k"
    if piece == B_QUEEN:
        return "q"
    if piece == B_ROOK:
        return "r"
    if piece == B_BISHOP:
        return "b"
    if piece == B_KNIGHT:
        return "n"
    if piece == B_PAWN:
        return "p"
    return ""


# ============================================================================
# ChessEnv
# ============================================================================


struct ChessEnv[DTYPE: DType = DType.float64](
    TwoPlayerDiscreteEnv & GPUTwoPlayerDiscreteEnv & RenderableEnv
):
    """Chess environment — CPU+GPU dual path, full rules.

    CPU: Instance methods for evaluation + single-agent mode.
    GPU: Static inline methods for batched self-play training.
    """

    comptime dtype = Self.DTYPE
    comptime StateType = BoardGameState
    comptime ActionType = BoardGameAction

    comptime STATE_SIZE: Int = 72
    comptime OBS_DIM: Int = 896  # 14 × 64
    comptime NUM_ACTIONS: Int = 4672  # 64 × 73

    var state: List[Scalar[Self.dtype]]
    var done: Bool

    # Renderer
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool
    var _sprite_pixels: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var _has_sprites: Bool

    def __init__(out self):
        self.state = List[Scalar[Self.dtype]](capacity=72)
        for _ in range(72):
            self.state.append(Scalar[Self.dtype](0.0))
        self.done = False
        self._renderer = None
        self._renderer_initialized = False
        self._sprite_pixels = None
        self._has_sprites = False

    # ========================================================================
    # Reset to initial position
    # ========================================================================

    def reset(mut self) -> BoardGameState:
        for i in range(72):
            self.state[i] = 0.0

        # White pieces (row 0 = rank 1)
        self.state[_sq(0, 0)] = Scalar[Self.dtype](W_ROOK)
        self.state[_sq(0, 1)] = Scalar[Self.dtype](W_KNIGHT)
        self.state[_sq(0, 2)] = Scalar[Self.dtype](W_BISHOP)
        self.state[_sq(0, 3)] = Scalar[Self.dtype](W_QUEEN)
        self.state[_sq(0, 4)] = Scalar[Self.dtype](W_KING)
        self.state[_sq(0, 5)] = Scalar[Self.dtype](W_BISHOP)
        self.state[_sq(0, 6)] = Scalar[Self.dtype](W_KNIGHT)
        self.state[_sq(0, 7)] = Scalar[Self.dtype](W_ROOK)
        for c in range(8):
            self.state[_sq(1, c)] = Scalar[Self.dtype](W_PAWN)

        # Black pieces (row 7 = rank 8)
        self.state[_sq(7, 0)] = Scalar[Self.dtype](B_ROOK)
        self.state[_sq(7, 1)] = Scalar[Self.dtype](B_KNIGHT)
        self.state[_sq(7, 2)] = Scalar[Self.dtype](B_BISHOP)
        self.state[_sq(7, 3)] = Scalar[Self.dtype](B_QUEEN)
        self.state[_sq(7, 4)] = Scalar[Self.dtype](B_KING)
        self.state[_sq(7, 5)] = Scalar[Self.dtype](B_BISHOP)
        self.state[_sq(7, 6)] = Scalar[Self.dtype](B_KNIGHT)
        self.state[_sq(7, 7)] = Scalar[Self.dtype](B_ROOK)
        for c in range(8):
            self.state[_sq(6, c)] = Scalar[Self.dtype](B_PAWN)

        self.state[S_PLAYER] = 0.0  # White to move
        self.state[S_CASTLING] = Scalar[Self.dtype](
            CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ
        )
        self.state[S_EP] = -1.0
        self.state[S_HALFMOVE] = 0.0
        self.state[S_FULLMOVE] = 1.0
        self.state[S_RESULT] = 0.0
        self.state[S_WK] = Scalar[Self.dtype](_sq(0, 4))
        self.state[S_BK] = Scalar[Self.dtype](_sq(7, 4))
        self.done = False
        return BoardGameState(index=0)

    # ========================================================================
    # Board access helpers
    # ========================================================================

    def _piece_at(self, sq: Int) -> Int:
        return Int(self.state[sq])

    def _player(self) -> Int:
        return Int(self.state[S_PLAYER])

    def _king_sq(self, player: Int) -> Int:
        if player == 0:
            return Int(self.state[S_WK])
        return Int(self.state[S_BK])

    # ========================================================================
    # Attack detection
    # ========================================================================

    def _is_attacked_by(self, sq: Int, attacker: Int) -> Bool:
        """Check if `sq` is attacked by `attacker` (0=white, 1=black)."""
        var r = _row(sq)
        var c = _col(sq)

        # Pawn attacks
        var pawn = _make_piece(1, attacker)  # pawn type = 1
        var pawn_dir = 1 if attacker == 0 else -1  # pawns attack forward
        # Pawns attack from (r - pawn_dir, c±1)
        var pr = r - pawn_dir
        if _on_board(pr, c - 1) and self._piece_at(_sq(pr, c - 1)) == pawn:
            return True
        if _on_board(pr, c + 1) and self._piece_at(_sq(pr, c + 1)) == pawn:
            return True

        # Knight attacks
        var knight = _make_piece(2, attacker)
        for k in range(8):
            var nr = r + _knight_dr(k)
            var nc = c + _knight_dc(k)
            if _on_board(nr, nc) and self._piece_at(_sq(nr, nc)) == knight:
                return True

        # Sliding attacks (bishop/rook/queen)
        var bishop = _make_piece(3, attacker)
        var rook = _make_piece(4, attacker)
        var queen = _make_piece(5, attacker)

        for d in range(8):
            var dr = _queen_dr(d)
            var dc = _queen_dc(d)
            var is_diag = dr != 0 and dc != 0
            var is_straight = not is_diag

            var sr = r + dr
            var sc = c + dc
            while _on_board(sr, sc):
                var p = self._piece_at(_sq(sr, sc))
                if p != EMPTY:
                    if p == queen:
                        return True
                    if is_diag and p == bishop:
                        return True
                    if is_straight and p == rook:
                        return True
                    break  # blocked
                sr += dr
                sc += dc

        # King attacks (adjacent)
        var king = _make_piece(6, attacker)
        for d in range(8):
            var kr = r + _queen_dr(d)
            var kc = c + _queen_dc(d)
            if _on_board(kr, kc) and self._piece_at(_sq(kr, kc)) == king:
                return True

        return False

    def _in_check(self, player: Int) -> Bool:
        """Check if `player`'s king is in check."""
        var ksq = self._king_sq(player)
        var opponent = 1 - player
        return self._is_attacked_by(ksq, opponent)

    # ========================================================================
    # Pseudo-legal move generation
    # ========================================================================

    def _gen_pseudo_legal(self, player: Int) -> List[Move]:
        """Generate all pseudo-legal moves for `player`.

        Does NOT filter moves that leave own king in check.
        """
        var moves = List[Move](capacity=128)
        # var pawn = _make_piece(1, player)
        # var knight = _make_piece(2, player)
        # var bishop = _make_piece(3, player)
        # var rook = _make_piece(4, player)
        # var queen = _make_piece(5, player)
        # var king = _make_piece(6, player)
        var fwd = 1 if player == 0 else -1
        var start_row = 1 if player == 0 else 6
        var promo_row = 7 if player == 0 else 0

        for sq in range(64):
            var p = self._piece_at(sq)
            if not _is_friendly(p, player):
                continue

            var r = _row(sq)
            var c = _col(sq)
            var pt = _piece_type(p)

            if pt == 1:  # Pawn
                # Forward 1
                var fr = r + fwd
                if _on_board(fr, c) and self._piece_at(_sq(fr, c)) == EMPTY:
                    if fr == promo_row:
                        moves.append(Move(_sq(r, c), _sq(fr, c), 5))  # queen
                        moves.append(Move(_sq(r, c), _sq(fr, c), 2))  # knight
                        moves.append(Move(_sq(r, c), _sq(fr, c), 3))  # bishop
                        moves.append(Move(_sq(r, c), _sq(fr, c), 4))  # rook
                    else:
                        moves.append(Move(_sq(r, c), _sq(fr, c)))
                    # Forward 2 from start
                    if r == start_row:
                        var fr2 = r + 2 * fwd
                        if self._piece_at(_sq(fr2, c)) == EMPTY:
                            moves.append(Move(_sq(r, c), _sq(fr2, c)))

                # Captures
                for dc in range(-1, 2, 2):  # -1, +1
                    var cr = r + fwd
                    var cc = c + dc
                    if not _on_board(cr, cc):
                        continue
                    var target = self._piece_at(_sq(cr, cc))
                    var ep_sq = Int(self.state[S_EP])
                    if _is_enemy(target, player) or _sq(cr, cc) == ep_sq:
                        if cr == promo_row:
                            moves.append(Move(_sq(r, c), _sq(cr, cc), 5))
                            moves.append(Move(_sq(r, c), _sq(cr, cc), 2))
                            moves.append(Move(_sq(r, c), _sq(cr, cc), 3))
                            moves.append(Move(_sq(r, c), _sq(cr, cc), 4))
                        else:
                            moves.append(Move(_sq(r, c), _sq(cr, cc)))

            elif pt == 2:  # Knight
                for k in range(8):
                    var nr = r + _knight_dr(k)
                    var nc = c + _knight_dc(k)
                    if _on_board(nr, nc):
                        var target = self._piece_at(_sq(nr, nc))
                        if not _is_friendly(target, player):
                            moves.append(Move(sq, _sq(nr, nc)))

            elif pt == 3 or pt == 4 or pt == 5:  # Bishop, Rook, Queen
                # var start_dir: Int
                # var end_dir: Int
                # if pt == 3:  # Bishop: diagonals only (1,3,5,7)
                #     start_dir = 1
                #     end_dir = 8
                # elif pt == 4:  # Rook: straights only (0,2,4,6)
                #     start_dir = 0
                #     end_dir = 8

                for d in range(8):
                    # Bishop: skip straights. Rook: skip diags.
                    var is_diag = _queen_dr(d) != 0 and _queen_dc(d) != 0
                    if pt == 3 and not is_diag:
                        continue
                    if pt == 4 and is_diag:
                        continue

                    var sr = r + _queen_dr(d)
                    var sc = c + _queen_dc(d)
                    while _on_board(sr, sc):
                        var target = self._piece_at(_sq(sr, sc))
                        if _is_friendly(target, player):
                            break
                        moves.append(Move(sq, _sq(sr, sc)))
                        if target != EMPTY:
                            break  # capture, stop sliding
                        sr += _queen_dr(d)
                        sc += _queen_dc(d)

            elif pt == 6:  # King
                for d in range(8):
                    var kr = r + _queen_dr(d)
                    var kc = c + _queen_dc(d)
                    if _on_board(kr, kc):
                        var target = self._piece_at(_sq(kr, kc))
                        if not _is_friendly(target, player):
                            moves.append(Move(sq, _sq(kr, kc)))

                # Castling
                var castling = Int(self.state[S_CASTLING])
                if player == 0:
                    # White kingside: e1→g1, rook h1→f1
                    if (castling & CASTLE_WK) != 0 and sq == _sq(0, 4):
                        if (
                            self._piece_at(_sq(0, 5)) == EMPTY
                            and self._piece_at(_sq(0, 6)) == EMPTY
                        ):
                            if (
                                not self._is_attacked_by(_sq(0, 4), 1)
                                and not self._is_attacked_by(_sq(0, 5), 1)
                                and not self._is_attacked_by(_sq(0, 6), 1)
                            ):
                                moves.append(Move(_sq(0, 4), _sq(0, 6)))
                    # White queenside: e1→c1, rook a1→d1
                    if (castling & CASTLE_WQ) != 0 and sq == _sq(0, 4):
                        if (
                            self._piece_at(_sq(0, 3)) == EMPTY
                            and self._piece_at(_sq(0, 2)) == EMPTY
                            and self._piece_at(_sq(0, 1)) == EMPTY
                        ):
                            if (
                                not self._is_attacked_by(_sq(0, 4), 1)
                                and not self._is_attacked_by(_sq(0, 3), 1)
                                and not self._is_attacked_by(_sq(0, 2), 1)
                            ):
                                moves.append(Move(_sq(0, 4), _sq(0, 2)))
                else:
                    # Black kingside: e8→g8
                    if (castling & CASTLE_BK) != 0 and sq == _sq(7, 4):
                        if (
                            self._piece_at(_sq(7, 5)) == EMPTY
                            and self._piece_at(_sq(7, 6)) == EMPTY
                        ):
                            if (
                                not self._is_attacked_by(_sq(7, 4), 0)
                                and not self._is_attacked_by(_sq(7, 5), 0)
                                and not self._is_attacked_by(_sq(7, 6), 0)
                            ):
                                moves.append(Move(_sq(7, 4), _sq(7, 6)))
                    # Black queenside: e8→c8
                    if (castling & CASTLE_BQ) != 0 and sq == _sq(7, 4):
                        if (
                            self._piece_at(_sq(7, 3)) == EMPTY
                            and self._piece_at(_sq(7, 2)) == EMPTY
                            and self._piece_at(_sq(7, 1)) == EMPTY
                        ):
                            if (
                                not self._is_attacked_by(_sq(7, 4), 0)
                                and not self._is_attacked_by(_sq(7, 3), 0)
                                and not self._is_attacked_by(_sq(7, 2), 0)
                            ):
                                moves.append(Move(_sq(7, 4), _sq(7, 2)))

        return moves^

    def _gen_legal_moves(mut self, player: Int) -> List[Move]:
        """Generate all legal moves (pseudo-legal + not leaving king in check).
        """
        var pseudo = self._gen_pseudo_legal(player)
        var legal = List[Move](capacity=len(pseudo))

        for i in range(len(pseudo)):
            var m = pseudo[i]
            # Save state
            var captured = self._piece_at(m.to_sq)
            var moved_piece = self._piece_at(m.from_sq)
            var old_ep = Int(self.state[S_EP])
            var old_wk = Int(self.state[S_WK])
            var old_bk = Int(self.state[S_BK])

            # Make move (simplified — just move piece)
            self.state[m.to_sq] = self.state[m.from_sq]
            self.state[m.from_sq] = 0.0

            # Handle en passant capture
            var ep_captured_sq = -1
            if _piece_type(moved_piece) == 1 and m.to_sq == old_ep:
                var ep_pawn_row = _row(m.to_sq) - (1 if player == 0 else -1)
                ep_captured_sq = _sq(ep_pawn_row, _col(m.to_sq))
                self.state[ep_captured_sq] = 0.0

            # Handle promotion
            if m.promo != 0:
                self.state[m.to_sq] = Scalar[Self.dtype](
                    _make_piece(m.promo, player)
                )

            # Update king position
            if _piece_type(moved_piece) == 6:
                if player == 0:
                    self.state[S_WK] = Scalar[Self.dtype](m.to_sq)
                else:
                    self.state[S_BK] = Scalar[Self.dtype](m.to_sq)

            # Check if own king is in check
            var in_check = self._in_check(player)

            # Unmake
            self.state[m.from_sq] = Scalar[Self.dtype](moved_piece)
            self.state[m.to_sq] = Scalar[Self.dtype](captured)
            if ep_captured_sq >= 0:
                var ep_pawn = _make_piece(1, 1 - player)
                self.state[ep_captured_sq] = Scalar[Self.dtype](ep_pawn)
            self.state[S_WK] = Scalar[Self.dtype](old_wk)
            self.state[S_BK] = Scalar[Self.dtype](old_bk)

            if not in_check:
                legal.append(m)

        return legal^

    # ========================================================================
    # Step
    # ========================================================================

    def step(
        mut self, action: BoardGameAction, verbose: Bool = False
    ) -> Tuple[BoardGameState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            BoardGameState(index=Int(self.state[S_FULLMOVE])),
            result[0],
            result[1],
        )

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Execute canonical action. Returns (reward, done)."""
        if self.done:
            return (Scalar[Self.dtype](0.0), True)

        var player = self._player()
        var m = _decode_action(action, player)

        # Validate action
        if m.to_sq < 0 or m.to_sq >= 64 or m.from_sq < 0 or m.from_sq >= 64:
            return (Scalar[Self.dtype](-1.0), False)

        var moved_piece = self._piece_at(m.from_sq)
        if not _is_friendly(moved_piece, player):
            return (Scalar[Self.dtype](-1.0), False)

        # Check legality by generating legal moves and matching
        var legal = self._gen_legal_moves(player)
        var found = False
        for i in range(len(legal)):
            if legal[i].from_sq == m.from_sq and legal[i].to_sq == m.to_sq:
                # Check promotion match
                if legal[i].promo == m.promo or (
                    legal[i].promo == 0 and m.promo == 0
                ):
                    # Use the legal move's promo if our decode defaulted to queen
                    if m.promo == 5 and legal[i].promo == 5:
                        found = True
                    elif m.promo == 0 and legal[i].promo == 0:
                        found = True
                    elif m.promo == legal[i].promo:
                        found = True
                    else:
                        found = False
                    if found:
                        break

        if not found:
            return (Scalar[Self.dtype](-1.0), False)

        # Execute move
        var captured = self._piece_at(m.to_sq)
        var is_pawn = _piece_type(moved_piece) == 1
        var is_capture = captured != EMPTY

        # En passant capture
        var old_ep = Int(self.state[S_EP])
        if is_pawn and m.to_sq == old_ep:
            var ep_pawn_row = _row(m.to_sq) - (1 if player == 0 else -1)
            self.state[_sq(ep_pawn_row, _col(m.to_sq))] = 0.0
            is_capture = True

        # Move piece
        self.state[m.to_sq] = self.state[m.from_sq]
        self.state[m.from_sq] = 0.0

        # Promotion
        if m.promo != 0:
            self.state[m.to_sq] = Scalar[Self.dtype](
                _make_piece(m.promo, player)
            )

        # Castling rook move
        if _piece_type(moved_piece) == 6:
            var from_c = _col(m.from_sq)
            var to_c = _col(m.to_sq)
            if to_c - from_c == 2:  # Kingside
                var rook_from = _sq(_row(m.from_sq), 7)
                var rook_to = _sq(_row(m.from_sq), 5)
                self.state[rook_to] = self.state[rook_from]
                self.state[rook_from] = 0.0
            elif from_c - to_c == 2:  # Queenside
                var rook_from = _sq(_row(m.from_sq), 0)
                var rook_to = _sq(_row(m.from_sq), 3)
                self.state[rook_to] = self.state[rook_from]
                self.state[rook_from] = 0.0

        # Update king position
        if _piece_type(moved_piece) == 6:
            if player == 0:
                self.state[S_WK] = Scalar[Self.dtype](m.to_sq)
            else:
                self.state[S_BK] = Scalar[Self.dtype](m.to_sq)

        # Update castling rights
        var castling = Int(self.state[S_CASTLING])
        if m.from_sq == _sq(0, 4):
            castling &= ~(CASTLE_WK | CASTLE_WQ)
        if m.from_sq == _sq(7, 4):
            castling &= ~(CASTLE_BK | CASTLE_BQ)
        if m.from_sq == _sq(0, 0) or m.to_sq == _sq(0, 0):
            castling &= ~CASTLE_WQ
        if m.from_sq == _sq(0, 7) or m.to_sq == _sq(0, 7):
            castling &= ~CASTLE_WK
        if m.from_sq == _sq(7, 0) or m.to_sq == _sq(7, 0):
            castling &= ~CASTLE_BQ
        if m.from_sq == _sq(7, 7) or m.to_sq == _sq(7, 7):
            castling &= ~CASTLE_BK
        self.state[S_CASTLING] = Scalar[Self.dtype](castling)

        # Update en passant
        if is_pawn and (
            _row(m.to_sq) - _row(m.from_sq) == 2
            or _row(m.from_sq) - _row(m.to_sq) == 2
        ):
            self.state[S_EP] = Scalar[Self.dtype](
                _sq((_row(m.from_sq) + _row(m.to_sq)) // 2, _col(m.from_sq))
            )
        else:
            self.state[S_EP] = -1.0

        # Update halfmove clock
        if is_pawn or is_capture:
            self.state[S_HALFMOVE] = 0.0
        else:
            self.state[S_HALFMOVE] += 1.0

        # Update fullmove
        if player == 1:
            self.state[S_FULLMOVE] += 1.0

        # Switch player
        var opponent = 1 - player
        self.state[S_PLAYER] = Scalar[Self.dtype](opponent)

        # Check game end
        var opp_legal = self._gen_legal_moves(opponent)
        var opp_in_check = self._in_check(opponent)

        if len(opp_legal) == 0:
            self.done = True
            if opp_in_check:
                # Checkmate — current player (who just moved) wins
                self.state[S_RESULT] = Scalar[Self.dtype](player + 1)
                return (Scalar[Self.dtype](1.0), True)
            else:
                # Stalemate
                self.state[S_RESULT] = Scalar[Self.dtype](RESULT_DRAW)
                return (Scalar[Self.dtype](0.0), True)

        # 50-move draw
        if Int(self.state[S_HALFMOVE]) >= 100:
            self.done = True
            self.state[S_RESULT] = Scalar[Self.dtype](RESULT_DRAW)
            return (Scalar[Self.dtype](0.0), True)

        return (Scalar[Self.dtype](0.0), False)

    # ========================================================================
    # Env trait methods
    # ========================================================================

    def get_state(mut self) -> BoardGameState:
        return BoardGameState(index=Int(self.state[S_FULLMOVE]))

    def close(mut self):
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    def action_from_index(self, action_idx: Int) -> BoardGameAction:
        return BoardGameAction(value=action_idx)

    def num_actions(self) -> Int:
        return 4672

    def obs_dim(self) -> Int:
        return 896

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: BoardGameState) -> Int:
        return state.index

    # ========================================================================
    # TwoPlayerDiscreteEnv trait methods
    # ========================================================================

    def current_player(self) -> Int:
        return self._player()

    def legal_action_mask(self) -> List[Bool]:
        var mask = List[Bool](capacity=4672)
        for _ in range(4672):
            mask.append(False)

        if self.done:
            return mask^

        var player = self._player()
        # Create a mutable copy to generate legal moves
        # (_gen_legal_moves does make/unmake which requires mut self)
        var env_copy = ChessEnv[Self.dtype]()
        for i in range(72):
            env_copy.state[i] = self.state[i]
        env_copy.done = self.done

        var legal = env_copy._gen_legal_moves(player)

        for i in range(len(legal)):
            var action = _encode_action(legal[i], player)
            if action >= 0 and action < 4672:
                mask[action] = True

        return mask^

    def game_result(self) -> Int:
        return Int(self.state[S_RESULT])

    # ========================================================================
    # Canonical observation
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=896)
        var player = self._player()

        # Piece type mapping for current player and opponent
        var my_offset = 0 if player == 0 else 6
        var opp_offset = 6 if player == 0 else 0

        # Planes 0-5: my pieces by type (pawn=1..king=6)
        for piece_type in range(1, 7):
            var target = piece_type + my_offset
            for sq in range(64):
                # var canonical_sq = sq if player == 0 else _flip_sq(sq)
                if self._piece_at(sq) == target:
                    obs.append(Scalar[Self.dtype](1.0))
                else:
                    obs.append(Scalar[Self.dtype](0.0))

        # Planes 6-11: opponent pieces by type
        for piece_type in range(1, 7):
            var target = piece_type + opp_offset
            for sq in range(64):
                if self._piece_at(sq) == target:
                    obs.append(Scalar[Self.dtype](1.0))
                else:
                    obs.append(Scalar[Self.dtype](0.0))

        # Plane 12: castling rights
        var castling = Int(self.state[S_CASTLING])
        var my_ks: Bool
        var my_qs: Bool
        if player == 0:
            my_ks = (castling & CASTLE_WK) != 0
            my_qs = (castling & CASTLE_WQ) != 0
        else:
            my_ks = (castling & CASTLE_BK) != 0
            my_qs = (castling & CASTLE_BQ) != 0

        for _ in range(64):
            if my_ks or my_qs:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 13: en passant
        var ep = Int(self.state[S_EP])
        for sq in range(64):
            var canonical_sq = sq if player == 0 else _flip_sq(sq)
            if ep >= 0 and canonical_sq == ep:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Single-agent step with random opponent."""
        var result = self._step_impl(action)
        var reward = result[0]
        var done = result[1]

        if done:
            return (self.get_obs_list(), reward, done)

        # Random opponent
        var opp_legal = self._gen_legal_moves(self._player())
        if len(opp_legal) > 0:
            var idx = Int(random_float64() * Float64(len(opp_legal)))
            if idx >= len(opp_legal):
                idx = len(opp_legal) - 1
            var opp_move = opp_legal[idx]
            var opp_action = _encode_action(opp_move, self._player())
            var opp_result = self._step_impl(opp_action)
            done = opp_result[1]
            if done:
                var gr = self.game_result()
                if gr == RESULT_WHITE_WINS:
                    reward = Scalar[Self.dtype](1.0)
                elif gr == RESULT_BLACK_WINS:
                    reward = Scalar[Self.dtype](-1.0)

        return (self.get_obs_list(), reward, done)

    # ========================================================================
    # RenderableEnv trait methods
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(width=536, height=586, fps=30, title="Chess")
        )
        self._renderer_initialized = True
        # Create sprite pixel data
        if not self._has_sprites:
            self._sprite_pixels = untracked(create_sprite_sheet())
            self._has_sprites = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def _piece_to_sprite_idx(self, piece: Int) -> Int:
        """Map piece ID (1-12) to sprite sheet index (0-11).

        Sprite order: wK(0) wQ(1) wR(2) wB(3) wN(4) wP(5) bK(6) bQ(7) bR(8) bB(9) bN(10) bP(11)
        Piece IDs: wP=1 wN=2 wB=3 wR=4 wQ=5 wK=6, bP=7 bN=8 bB=9 bR=10 bQ=11 bK=12
        """
        if piece == W_KING:
            return 0
        if piece == W_QUEEN:
            return 1
        if piece == W_ROOK:
            return 2
        if piece == W_BISHOP:
            return 3
        if piece == W_KNIGHT:
            return 4
        if piece == W_PAWN:
            return 5
        if piece == B_KING:
            return 6
        if piece == B_QUEEN:
            return 7
        if piece == B_ROOK:
            return 8
        if piece == B_BISHOP:
            return 9
        if piece == B_KNIGHT:
            return 10
        if piece == B_PAWN:
            return 11
        return -1

    def _render(self, mut renderer: Renderer2D):
        """Render chess board state using SDL3 with sprite pieces."""
        from noeira.render.sdl import (
            create_surface_from,
            create_texture_from_surface,
            render_texture,
            destroy_surface,
            set_texture_blend_mode,
            set_texture_scale_mode,
            destroy_texture,
            Surface,
            Texture,
            FRect,
            PixelFormat,
            BlendMode,
            ScaleMode,
        )

        var bg_color = SDL_Color(r=0x22, g=0x22, b=0x22, a=0xFF)
        if not renderer.begin_frame_with_color(bg_color):
            return

        var light_sq = SDL_Color(r=0xF0, g=0xD9, b=0xB5, a=0xFF)
        var dark_sq = SDL_Color(r=0xB5, g=0x88, b=0x63, a=0xFF)
        var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
        var status_bg_color = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)

        var left_margin = 20
        var top_margin = 4
        var sq_size = 64
        var board_px = sq_size * 8  # 512
        var sprite_draw_size = 48  # Scale 24→48 to fit in 64px cells
        var sprite_offset = (sq_size - sprite_draw_size) // 2  # Center in cell

        # Create texture from sprite pixels (recreated each frame for simplicity)
        var has_texture = False
        var texture: Optional[
            Pointer[Texture, MutAnyOrigin]
        ] = None
        if self._has_sprites:
            try:
                var surface = create_surface_from(
                    c_int(SPRITE_SHEET_WIDTH),
                    c_int(SPRITE_SHEET_HEIGHT),
                    PixelFormat.PIXELFORMAT_RGBA32,
                    rebind[Pointer[NoneType, MutAnyOrigin]](
                        self._sprite_pixels
                    ),
                    c_int(SPRITE_SHEET_WIDTH * SPRITE_BPP),
                )
                texture = create_texture_from_surface(
                    renderer.sdl_renderer.value(), surface
                )
                set_texture_blend_mode(texture.value(), BlendMode.BLENDMODE_BLEND)
                try:
                    set_texture_scale_mode(texture.value(), ScaleMode.SCALEMODE_NEAREST)
                except:
                    pass
                destroy_surface(surface)
                has_texture = True
            except:
                pass

        # Draw board squares and pieces
        for row in range(8):
            for col in range(8):
                var px = left_margin + col * sq_size
                var py = top_margin + row * sq_size

                # Square color
                var is_light = ((row + col) % 2) == 0
                if is_light:
                    renderer.draw_rect(px, py, sq_size, sq_size, light_sq)
                else:
                    renderer.draw_rect(px, py, sq_size, sq_size, dark_sq)

                # Env coords: display row 0 = env row 7
                var env_row = 7 - row
                var env_sq = env_row * 8 + col
                var piece = Int(self.state[env_sq])

                if piece != EMPTY and has_texture:
                    # Draw sprite
                    var sprite_idx = self._piece_to_sprite_idx(piece)
                    if sprite_idx >= 0:
                        var src_rect_alloc = alloc[FRect]({count = 1})
                        var src_rect = src_rect_alloc.unsafe_ptr()
                        src_rect[] = FRect(
                            c_float(sprite_idx * SPRITE_SIZE),
                            c_float(0),
                            c_float(SPRITE_SIZE),
                            c_float(SPRITE_SIZE),
                        )
                        var dst_rect_alloc = alloc[FRect]({count = 1})
                        var dst_rect = dst_rect_alloc.unsafe_ptr()
                        dst_rect[] = FRect(
                            c_float(px + sprite_offset),
                            c_float(py + sprite_offset),
                            c_float(sprite_draw_size),
                            c_float(sprite_draw_size),
                        )
                        try:
                            render_texture(
                                renderer.sdl_renderer.value(),
                                texture.value(),
                                rebind[Pointer[FRect, ImmutAnyOrigin]](
                                    src_rect
                                ),
                                rebind[Pointer[FRect, ImmutAnyOrigin]](
                                    dst_rect
                                ),
                            )
                        except:
                            pass
                        dealloc(src_rect_alloc^)
                        dealloc(dst_rect_alloc^)
                elif piece != EMPTY:
                    # Fallback: draw text
                    var pc = _piece_char(piece)
                    var tx = px + sq_size // 2 - 4
                    var ty = py + sq_size // 2 - 4
                    renderer.draw_text(pc, tx, ty, text_color)

        # Clean up texture
        if has_texture:
            try:
                destroy_texture(texture.value())
            except:
                pass

        # File labels (a-h) below the board
        var label_color = SDL_Color(r=0xCC, g=0xCC, b=0xCC, a=0xFF)
        for col in range(8):
            var lx = left_margin + col * sq_size + sq_size // 2 - 4
            var ly = top_margin + 8 * sq_size + 4
            var fl: String
            if col == 0:
                fl = "a"
            elif col == 1:
                fl = "b"
            elif col == 2:
                fl = "c"
            elif col == 3:
                fl = "d"
            elif col == 4:
                fl = "e"
            elif col == 5:
                fl = "f"
            elif col == 6:
                fl = "g"
            else:
                fl = "h"
            renderer.draw_text(fl, lx, ly, label_color)

        # Rank labels (8 down to 1) to the left of the board
        for display_row in range(8):
            var rank_num = 8 - display_row
            var lx = left_margin - 12
            var ly = top_margin + display_row * sq_size + sq_size // 2 - 4
            renderer.draw_text(String(rank_num), lx, ly, label_color)

        # Status bar
        var status_y = top_margin + board_px + 20
        renderer.draw_rect(0, status_y, 536, 50, status_bg_color)

        var result = Int(self.state[S_RESULT])
        var player = self._player()
        if result == RESULT_ONGOING:
            if player == 0:
                renderer.draw_text(
                    "White's turn", 200, status_y + 20, text_color
                )
            else:
                renderer.draw_text(
                    "Black's turn", 200, status_y + 20, text_color
                )
        elif result == RESULT_WHITE_WINS:
            renderer.draw_text(
                "Checkmate! White wins", 150, status_y + 20, text_color
            )
        elif result == RESULT_BLACK_WINS:
            renderer.draw_text(
                "Checkmate! Black wins", 150, status_y + 20, text_color
            )
        elif result == RESULT_DRAW:
            renderer.draw_text("Draw!", 220, status_y + 20, text_color)

        renderer.flip()

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False
        if self._has_sprites:
            self._sprite_pixels.value().unsafe_free()
            self._has_sprites = False

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU: Inline helper methods
    # ========================================================================

    comptime TPB = 256

    @staticmethod
    @always_inline
    def _gpu_is_attacked_by[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        g: Int,
        sq: Int,
        attacker: Int,
    ) -> Bool:
        """Check if `sq` is attacked by `attacker` (0=white, 1=black) on GPU."""
        var r = _row(sq)
        var c = _col(sq)

        # Pawn attacks
        var pawn = 1 + attacker * 6
        var pawn_dir = 1 if attacker == 0 else -1
        var pr = r - pawn_dir
        if _on_board(pr, c - 1) and Int(states[g, _sq(pr, c - 1)]) == pawn:
            return True
        if _on_board(pr, c + 1) and Int(states[g, _sq(pr, c + 1)]) == pawn:
            return True

        # Knight attacks
        var knight = 2 + attacker * 6
        for k in range(8):
            var nr = r + _knight_dr(k)
            var nc = c + _knight_dc(k)
            if _on_board(nr, nc) and Int(states[g, _sq(nr, nc)]) == knight:
                return True

        # Sliding attacks
        var bishop = 3 + attacker * 6
        var rook = 4 + attacker * 6
        var queen = 5 + attacker * 6
        for d in range(8):
            var dr = _queen_dr(d)
            var dc = _queen_dc(d)
            var is_diag = dr != 0 and dc != 0
            var sr = r + dr
            var sc = c + dc
            while _on_board(sr, sc):
                var p = Int(states[g, _sq(sr, sc)])
                if p != EMPTY:
                    if p == queen:
                        return True
                    if is_diag and p == bishop:
                        return True
                    if not is_diag and p == rook:
                        return True
                    break
                sr += dr
                sc += dc

        # King attacks
        var king = 6 + attacker * 6
        for d in range(8):
            var kr = r + _queen_dr(d)
            var kc = c + _queen_dc(d)
            if _on_board(kr, kc) and Int(states[g, _sq(kr, kc)]) == king:
                return True

        return False

    @staticmethod
    @always_inline
    def _gpu_in_check[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        g: Int,
        player: Int,
    ) -> Bool:
        var ksq: Int
        if player == 0:
            ksq = Int(states[g, S_WK])
        else:
            ksq = Int(states[g, S_BK])
        return ChessEnv._gpu_is_attacked_by[BATCH_SIZE, STATE_SIZE](
            states, g, ksq, 1 - player
        )

    @staticmethod
    @always_inline
    def _gpu_try_move_legal[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        g: Int,
        from_sq: Int,
        to_sq: Int,
        promo: Int,
        player: Int,
    ) -> Bool:
        """Make move, check if own king is safe, unmake. Returns True if legal.
        """
        var moved = Int(states[g, from_sq])
        var captured = Int(states[g, to_sq])
        var old_wk = Int(states[g, S_WK])
        var old_bk = Int(states[g, S_BK])
        var old_ep = Int(states[g, S_EP])

        states[g, to_sq] = states[g, from_sq]
        states[g, from_sq] = 0.0

        var ep_sq = -1
        var ep_piece = 0
        if _piece_type(moved) == 1 and to_sq == old_ep:
            var ep_r = _row(to_sq) - (1 if player == 0 else -1)
            ep_sq = _sq(ep_r, _col(to_sq))
            ep_piece = Int(states[g, ep_sq])
            states[g, ep_sq] = 0.0

        if promo != 0:
            states[g, to_sq] = Scalar[board_dtype](_make_piece(promo, player))

        if _piece_type(moved) == 6:
            if player == 0:
                states[g, S_WK] = Scalar[board_dtype](to_sq)
            else:
                states[g, S_BK] = Scalar[board_dtype](to_sq)

        var legal = not ChessEnv._gpu_in_check[BATCH_SIZE, STATE_SIZE](
            states, g, player
        )

        states[g, from_sq] = Scalar[board_dtype](moved)
        states[g, to_sq] = Scalar[board_dtype](captured)
        if ep_sq >= 0:
            states[g, ep_sq] = Scalar[board_dtype](ep_piece)
        states[g, S_WK] = Scalar[board_dtype](old_wk)
        states[g, S_BK] = Scalar[board_dtype](old_bk)

        return legal

    @staticmethod
    @always_inline
    def _gpu_gen_legal_mask_and_count[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        NUM_ACTIONS: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        legal_masks: Tile[board_dtype, BATCH_SIZE, NUM_ACTIONS],
        g: Int,
        player: Int,
    ) -> Int:
        """Generate legal mask for all 4672 actions. Returns count of legal moves.
        """
        for a in range(4672):
            legal_masks[g, a] = 0.0

        var fwd = 1 if player == 0 else -1
        var start_row = 1 if player == 0 else 6
        var promo_row = 7 if player == 0 else 0
        var count = 0

        for sq in range(64):
            var p = Int(states[g, sq])
            if p == EMPTY:
                continue
            if player == 0 and not _is_white(p):
                continue
            if player == 1 and not _is_black(p):
                continue

            var r = _row(sq)
            var c = _col(sq)
            var pt = _piece_type(p)

            if pt == 1:  # Pawn
                var fr = r + fwd
                if _on_board(fr, c) and Int(states[g, _sq(fr, c)]) == EMPTY:
                    if fr == promo_row:
                        for promo in range(2, 6):
                            if ChessEnv._gpu_try_move_legal[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, sq, _sq(fr, c), promo, player):
                                var action = _encode_action(
                                    Move(sq, _sq(fr, c), promo), player
                                )
                                if action >= 0 and action < 4672:
                                    legal_masks[g, action] = 1.0
                                    count += 1
                    else:
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, g, sq, _sq(fr, c), 0, player
                        ):
                            var action = _encode_action(
                                Move(sq, _sq(fr, c), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1
                        if r == start_row:
                            var fr2 = r + 2 * fwd
                            if Int(states[g, _sq(fr2, c)]) == EMPTY:
                                if ChessEnv._gpu_try_move_legal[
                                    BATCH_SIZE, STATE_SIZE
                                ](states, g, sq, _sq(fr2, c), 0, player):
                                    var action = _encode_action(
                                        Move(sq, _sq(fr2, c), 0), player
                                    )
                                    if action >= 0 and action < 4672:
                                        legal_masks[g, action] = 1.0
                                        count += 1

                for dc_idx in range(2):
                    var dc = -1 if dc_idx == 0 else 1
                    var cr2 = r + fwd
                    var cc = c + dc
                    if not _on_board(cr2, cc):
                        continue
                    var target = Int(states[g, _sq(cr2, cc)])
                    var ep = Int(states[g, S_EP])
                    if _is_enemy(target, player) or _sq(cr2, cc) == ep:
                        if cr2 == promo_row:
                            for promo in range(2, 6):
                                if ChessEnv._gpu_try_move_legal[
                                    BATCH_SIZE, STATE_SIZE
                                ](states, g, sq, _sq(cr2, cc), promo, player):
                                    var action = _encode_action(
                                        Move(sq, _sq(cr2, cc), promo), player
                                    )
                                    if action >= 0 and action < 4672:
                                        legal_masks[g, action] = 1.0
                                        count += 1
                        else:
                            if ChessEnv._gpu_try_move_legal[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, sq, _sq(cr2, cc), 0, player):
                                var action = _encode_action(
                                    Move(sq, _sq(cr2, cc), 0), player
                                )
                                if action >= 0 and action < 4672:
                                    legal_masks[g, action] = 1.0
                                    count += 1

            elif pt == 2:  # Knight
                for k in range(8):
                    var nr = r + _knight_dr(k)
                    var nc = c + _knight_dc(k)
                    if _on_board(nr, nc) and not _is_friendly(
                        Int(states[g, _sq(nr, nc)]), player
                    ):
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, g, sq, _sq(nr, nc), 0, player
                        ):
                            var action = _encode_action(
                                Move(sq, _sq(nr, nc), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1

            elif pt == 3 or pt == 4 or pt == 5:
                for d in range(8):
                    var is_diag = _queen_dr(d) != 0 and _queen_dc(d) != 0
                    if pt == 3 and not is_diag:
                        continue
                    if pt == 4 and is_diag:
                        continue
                    var sr = r + _queen_dr(d)
                    var sc = c + _queen_dc(d)
                    while _on_board(sr, sc):
                        var target = Int(states[g, _sq(sr, sc)])
                        if _is_friendly(target, player):
                            break
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, g, sq, _sq(sr, sc), 0, player
                        ):
                            var action = _encode_action(
                                Move(sq, _sq(sr, sc), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1
                        if target != EMPTY:
                            break
                        sr += _queen_dr(d)
                        sc += _queen_dc(d)

            elif pt == 6:  # King
                for d in range(8):
                    var kr = r + _queen_dr(d)
                    var kc = c + _queen_dc(d)
                    if _on_board(kr, kc) and not _is_friendly(
                        Int(states[g, _sq(kr, kc)]), player
                    ):
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, g, sq, _sq(kr, kc), 0, player
                        ):
                            var action = _encode_action(
                                Move(sq, _sq(kr, kc), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1

                # Castling
                var castling = Int(states[g, S_CASTLING])
                if player == 0 and sq == _sq(0, 4):
                    if (
                        (castling & CASTLE_WK) != 0
                        and Int(states[g, _sq(0, 5)]) == EMPTY
                        and Int(states[g, _sq(0, 6)]) == EMPTY
                    ):
                        if (
                            not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(0, 4), 1)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(0, 5), 1)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(0, 6), 1)
                        ):
                            var action = _encode_action(
                                Move(_sq(0, 4), _sq(0, 6), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1
                    if (
                        (castling & CASTLE_WQ) != 0
                        and Int(states[g, _sq(0, 3)]) == EMPTY
                        and Int(states[g, _sq(0, 2)]) == EMPTY
                        and Int(states[g, _sq(0, 1)]) == EMPTY
                    ):
                        if (
                            not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(0, 4), 1)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(0, 3), 1)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(0, 2), 1)
                        ):
                            var action = _encode_action(
                                Move(_sq(0, 4), _sq(0, 2), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1
                elif player == 1 and sq == _sq(7, 4):
                    if (
                        (castling & CASTLE_BK) != 0
                        and Int(states[g, _sq(7, 5)]) == EMPTY
                        and Int(states[g, _sq(7, 6)]) == EMPTY
                    ):
                        if (
                            not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(7, 4), 0)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(7, 5), 0)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(7, 6), 0)
                        ):
                            var action = _encode_action(
                                Move(_sq(7, 4), _sq(7, 6), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1
                    if (
                        (castling & CASTLE_BQ) != 0
                        and Int(states[g, _sq(7, 3)]) == EMPTY
                        and Int(states[g, _sq(7, 2)]) == EMPTY
                        and Int(states[g, _sq(7, 1)]) == EMPTY
                    ):
                        if (
                            not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(7, 4), 0)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(7, 3), 0)
                            and not ChessEnv._gpu_is_attacked_by[
                                BATCH_SIZE, STATE_SIZE
                            ](states, g, _sq(7, 2), 0)
                        ):
                            var action = _encode_action(
                                Move(_sq(7, 4), _sq(7, 2), 0), player
                            )
                            if action >= 0 and action < 4672:
                                legal_masks[g, action] = 1.0
                                count += 1

        return count

    # ========================================================================
    # GPU: Step kernel
    # ========================================================================

    @staticmethod
    @always_inline
    def step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        actions: ImmTile[board_dtype, BATCH_SIZE],
        rewards: Tile[board_dtype, BATCH_SIZE],
        dones: Tile[board_dtype, BATCH_SIZE],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        if states[i, S_RESULT] != 0.0:
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        var action = Int(actions[i])
        var player = Int(states[i, S_PLAYER])
        var m = _decode_action(action, player)

        if m.to_sq < 0 or m.to_sq >= 64 or m.from_sq < 0 or m.from_sq >= 64:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        var moved = Int(states[i, m.from_sq])
        if (
            moved == EMPTY
            or (player == 0 and not _is_white(moved))
            or (player == 1 and not _is_black(moved))
        ):
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Pseudo-legality check: verify piece can make this move pattern
        var fr = _row(m.from_sq)
        var fc = _col(m.from_sq)
        var tr = _row(m.to_sq)
        var tc = _col(m.to_sq)
        var dr = tr - fr
        var dc = tc - fc
        var abs_dr = dr if dr >= 0 else -dr
        var abs_dc = dc if dc >= 0 else -dc
        var pt = _piece_type(moved)
        var fwd = 1 if player == 0 else -1
        var pseudo_legal = False

        if pt == 1:  # Pawn
            if dc == 0 and dr == fwd and Int(states[i, m.to_sq]) == EMPTY:
                pseudo_legal = True
            elif dc == 0 and dr == 2 * fwd and fr == (1 if player == 0 else 6):
                if (
                    Int(states[i, _sq(fr + fwd, fc)]) == EMPTY
                    and Int(states[i, m.to_sq]) == EMPTY
                ):
                    pseudo_legal = True
            elif abs_dc == 1 and dr == fwd:
                if _is_enemy(Int(states[i, m.to_sq]), player) or m.to_sq == Int(
                    states[i, S_EP]
                ):
                    pseudo_legal = True
        elif pt == 2:  # Knight
            if (abs_dr == 2 and abs_dc == 1) or (abs_dr == 1 and abs_dc == 2):
                if not _is_friendly(Int(states[i, m.to_sq]), player):
                    pseudo_legal = True
        elif pt == 3:  # Bishop (diagonal only)
            if abs_dr == abs_dc and abs_dr > 0:
                pseudo_legal = True  # path check done implicitly by legal mask
        elif pt == 4:  # Rook (straight only)
            if (abs_dr == 0 or abs_dc == 0) and (abs_dr + abs_dc) > 0:
                pseudo_legal = True
        elif pt == 5:  # Queen (diagonal or straight)
            if (abs_dr == abs_dc and abs_dr > 0) or (
                (abs_dr == 0 or abs_dc == 0) and (abs_dr + abs_dc) > 0
            ):
                pseudo_legal = True
        elif pt == 6:  # King
            if abs_dr <= 1 and abs_dc <= 1 and (abs_dr + abs_dc) > 0:
                pseudo_legal = True
            elif abs_dc == 2 and dr == 0:  # Castling
                pseudo_legal = True

        if not pseudo_legal:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Target must not be friendly
        if _is_friendly(Int(states[i, m.to_sq]), player) and pt != 1:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Path must be clear for sliding pieces
        if (pt == 3 or pt == 4 or pt == 5) and max(abs_dr, abs_dc) > 1:
            var step_r = 0
            if dr > 0:
                step_r = 1
            elif dr < 0:
                step_r = -1
            var step_c = 0
            if dc > 0:
                step_c = 1
            elif dc < 0:
                step_c = -1
            var cr = fr + step_r
            var cc = fc + step_c
            var blocked = False
            while cr != tr or cc != tc:
                if Int(states[i, _sq(cr, cc)]) != EMPTY:
                    blocked = True
                    break
                cr += step_r
                cc += step_c
            if blocked:
                rewards[i] = -1.0
                dones[i] = 0.0
                return

        if not ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
            states, i, m.from_sq, m.to_sq, m.promo, player
        ):
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Execute move
        var is_pawn = _piece_type(moved) == 1
        var captured = Int(states[i, m.to_sq])
        var is_capture = captured != EMPTY

        var old_ep = Int(states[i, S_EP])
        if is_pawn and m.to_sq == old_ep:
            var ep_r = _row(m.to_sq) - (1 if player == 0 else -1)
            states[i, _sq(ep_r, _col(m.to_sq))] = 0.0
            is_capture = True

        states[i, m.to_sq] = states[i, m.from_sq]
        states[i, m.from_sq] = 0.0

        if m.promo != 0:
            states[i, m.to_sq] = Scalar[board_dtype](
                _make_piece(m.promo, player)
            )

        if _piece_type(moved) == 6:
            var fc = _col(m.from_sq)
            var tc = _col(m.to_sq)
            if tc - fc == 2:
                var rr = _row(m.from_sq)
                states[i, _sq(rr, 5)] = states[i, _sq(rr, 7)]
                states[i, _sq(rr, 7)] = 0.0
            elif fc - tc == 2:
                var rr = _row(m.from_sq)
                states[i, _sq(rr, 3)] = states[i, _sq(rr, 0)]
                states[i, _sq(rr, 0)] = 0.0
            if player == 0:
                states[i, S_WK] = Scalar[board_dtype](m.to_sq)
            else:
                states[i, S_BK] = Scalar[board_dtype](m.to_sq)

        var cr = Int(states[i, S_CASTLING])
        if m.from_sq == _sq(0, 4):
            cr &= ~(CASTLE_WK | CASTLE_WQ)
        if m.from_sq == _sq(7, 4):
            cr &= ~(CASTLE_BK | CASTLE_BQ)
        if m.from_sq == _sq(0, 0) or m.to_sq == _sq(0, 0):
            cr &= ~CASTLE_WQ
        if m.from_sq == _sq(0, 7) or m.to_sq == _sq(0, 7):
            cr &= ~CASTLE_WK
        if m.from_sq == _sq(7, 0) or m.to_sq == _sq(7, 0):
            cr &= ~CASTLE_BQ
        if m.from_sq == _sq(7, 7) or m.to_sq == _sq(7, 7):
            cr &= ~CASTLE_BK
        states[i, S_CASTLING] = Scalar[board_dtype](cr)

        if is_pawn and (
            _row(m.to_sq) - _row(m.from_sq) == 2
            or _row(m.from_sq) - _row(m.to_sq) == 2
        ):
            states[i, S_EP] = Scalar[board_dtype](
                _sq((_row(m.from_sq) + _row(m.to_sq)) // 2, _col(m.from_sq))
            )
        else:
            states[i, S_EP] = -1.0

        if is_pawn or is_capture:
            states[i, S_HALFMOVE] = 0.0
        else:
            states[i, S_HALFMOVE] = states[i, S_HALFMOVE] + 1.0

        if player == 1:
            states[i, S_FULLMOVE] = states[i, S_FULLMOVE] + 1.0

        var opponent = 1 - player
        states[i, S_PLAYER] = Scalar[board_dtype](opponent)

        # Check game end — find if opponent has any legal move
        var opp_has_move = False
        var opp_in_check = ChessEnv._gpu_in_check[BATCH_SIZE, STATE_SIZE](
            states, i, opponent
        )
        var opp_fwd = 1 if opponent == 0 else -1
        var opp_start = 1 if opponent == 0 else 6
        var opp_promo = 7 if opponent == 0 else 0

        for sq in range(64):
            if opp_has_move:
                break
            var p = Int(states[i, sq])
            if p == EMPTY:
                continue
            if opponent == 0 and not _is_white(p):
                continue
            if opponent == 1 and not _is_black(p):
                continue
            var r = _row(sq)
            var c = _col(sq)
            var pt = _piece_type(p)

            if pt == 1:
                var fr = r + opp_fwd
                if _on_board(fr, c) and Int(states[i, _sq(fr, c)]) == EMPTY:
                    var pr2 = 5 if fr == opp_promo else 0
                    if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                        states, i, sq, _sq(fr, c), pr2, opponent
                    ):
                        opp_has_move = True
                if not opp_has_move and r == opp_start:
                    var fr2 = r + 2 * opp_fwd
                    if (
                        Int(states[i, _sq(r + opp_fwd, c)]) == EMPTY
                        and Int(states[i, _sq(fr2, c)]) == EMPTY
                    ):
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, i, sq, _sq(fr2, c), 0, opponent
                        ):
                            opp_has_move = True
                if not opp_has_move:
                    for dc_idx in range(2):
                        var dc = -1 if dc_idx == 0 else 1
                        var cc = c + dc
                        var cr2 = r + opp_fwd
                        if _on_board(cr2, cc):
                            var t = Int(states[i, _sq(cr2, cc)])
                            if _is_enemy(t, opponent) or _sq(cr2, cc) == Int(
                                states[i, S_EP]
                            ):
                                var pr2 = 5 if cr2 == opp_promo else 0
                                if ChessEnv._gpu_try_move_legal[
                                    BATCH_SIZE, STATE_SIZE
                                ](states, i, sq, _sq(cr2, cc), pr2, opponent):
                                    opp_has_move = True
                                    break
            elif pt == 2:
                for k in range(8):
                    var nr = r + _knight_dr(k)
                    var nc = c + _knight_dc(k)
                    if _on_board(nr, nc) and not _is_friendly(
                        Int(states[i, _sq(nr, nc)]), opponent
                    ):
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, i, sq, _sq(nr, nc), 0, opponent
                        ):
                            opp_has_move = True
                            break
            elif pt == 3 or pt == 4 or pt == 5:
                for d in range(8):
                    if opp_has_move:
                        break
                    var is_diag = _queen_dr(d) != 0 and _queen_dc(d) != 0
                    if pt == 3 and not is_diag:
                        continue
                    if pt == 4 and is_diag:
                        continue
                    var sr = r + _queen_dr(d)
                    var sc = c + _queen_dc(d)
                    while _on_board(sr, sc):
                        var t = Int(states[i, _sq(sr, sc)])
                        if _is_friendly(t, opponent):
                            break
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, i, sq, _sq(sr, sc), 0, opponent
                        ):
                            opp_has_move = True
                            break
                        if t != EMPTY:
                            break
                        sr += _queen_dr(d)
                        sc += _queen_dc(d)
            elif pt == 6:
                for d in range(8):
                    var kr = r + _queen_dr(d)
                    var kc = c + _queen_dc(d)
                    if _on_board(kr, kc) and not _is_friendly(
                        Int(states[i, _sq(kr, kc)]), opponent
                    ):
                        if ChessEnv._gpu_try_move_legal[BATCH_SIZE, STATE_SIZE](
                            states, i, sq, _sq(kr, kc), 0, opponent
                        ):
                            opp_has_move = True
                            break

        if not opp_has_move:
            if opp_in_check:
                states[i, S_RESULT] = Scalar[board_dtype](player + 1)
                rewards[i] = 1.0
            else:
                states[i, S_RESULT] = Scalar[board_dtype](RESULT_DRAW)
                rewards[i] = 0.0
            dones[i] = 1.0
            return

        if Int(states[i, S_HALFMOVE]) >= 100:
            states[i, S_RESULT] = Scalar[board_dtype](RESULT_DRAW)
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        rewards[i] = 0.0
        dones[i] = 0.0

    # ========================================================================
    # GPU: Reset / selective reset / obs extraction
    # ========================================================================

    @staticmethod
    @always_inline
    def reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        for c in range(72):
            states[i, c] = 0.0
        states[i, _sq(0, 0)] = Scalar[board_dtype](W_ROOK)
        states[i, _sq(0, 1)] = Scalar[board_dtype](W_KNIGHT)
        states[i, _sq(0, 2)] = Scalar[board_dtype](W_BISHOP)
        states[i, _sq(0, 3)] = Scalar[board_dtype](W_QUEEN)
        states[i, _sq(0, 4)] = Scalar[board_dtype](W_KING)
        states[i, _sq(0, 5)] = Scalar[board_dtype](W_BISHOP)
        states[i, _sq(0, 6)] = Scalar[board_dtype](W_KNIGHT)
        states[i, _sq(0, 7)] = Scalar[board_dtype](W_ROOK)
        for c in range(8):
            states[i, _sq(1, c)] = Scalar[board_dtype](W_PAWN)
        states[i, _sq(7, 0)] = Scalar[board_dtype](B_ROOK)
        states[i, _sq(7, 1)] = Scalar[board_dtype](B_KNIGHT)
        states[i, _sq(7, 2)] = Scalar[board_dtype](B_BISHOP)
        states[i, _sq(7, 3)] = Scalar[board_dtype](B_QUEEN)
        states[i, _sq(7, 4)] = Scalar[board_dtype](B_KING)
        states[i, _sq(7, 5)] = Scalar[board_dtype](B_BISHOP)
        states[i, _sq(7, 6)] = Scalar[board_dtype](B_KNIGHT)
        states[i, _sq(7, 7)] = Scalar[board_dtype](B_ROOK)
        for c in range(8):
            states[i, _sq(6, c)] = Scalar[board_dtype](B_PAWN)
        states[i, S_CASTLING] = Scalar[board_dtype](
            CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ
        )
        states[i, S_EP] = -1.0
        states[i, S_FULLMOVE] = 1.0
        states[i, S_WK] = Scalar[board_dtype](_sq(0, 4))
        states[i, S_BK] = Scalar[board_dtype](_sq(7, 4))

    @staticmethod
    @always_inline
    def selective_reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        dones: Tile[board_dtype, BATCH_SIZE],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        if dones[i] > 0.5:
            # Inline reset for this thread
            for c in range(72):
                states[i, c] = 0.0
            states[i, _sq(0, 0)] = Scalar[board_dtype](W_ROOK)
            states[i, _sq(0, 1)] = Scalar[board_dtype](W_KNIGHT)
            states[i, _sq(0, 2)] = Scalar[board_dtype](W_BISHOP)
            states[i, _sq(0, 3)] = Scalar[board_dtype](W_QUEEN)
            states[i, _sq(0, 4)] = Scalar[board_dtype](W_KING)
            states[i, _sq(0, 5)] = Scalar[board_dtype](W_BISHOP)
            states[i, _sq(0, 6)] = Scalar[board_dtype](W_KNIGHT)
            states[i, _sq(0, 7)] = Scalar[board_dtype](W_ROOK)
            for c in range(8):
                states[i, _sq(1, c)] = Scalar[board_dtype](W_PAWN)
            states[i, _sq(7, 0)] = Scalar[board_dtype](B_ROOK)
            states[i, _sq(7, 1)] = Scalar[board_dtype](B_KNIGHT)
            states[i, _sq(7, 2)] = Scalar[board_dtype](B_BISHOP)
            states[i, _sq(7, 3)] = Scalar[board_dtype](B_QUEEN)
            states[i, _sq(7, 4)] = Scalar[board_dtype](B_KING)
            states[i, _sq(7, 5)] = Scalar[board_dtype](B_BISHOP)
            states[i, _sq(7, 6)] = Scalar[board_dtype](B_KNIGHT)
            states[i, _sq(7, 7)] = Scalar[board_dtype](B_ROOK)
            for c in range(8):
                states[i, _sq(6, c)] = Scalar[board_dtype](B_PAWN)
            states[i, S_CASTLING] = Scalar[board_dtype](
                CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ
            )
            states[i, S_EP] = -1.0
            states[i, S_FULLMOVE] = 1.0
            states[i, S_WK] = Scalar[board_dtype](_sq(0, 4))
            states[i, S_BK] = Scalar[board_dtype](_sq(7, 4))
            dones[i] = 0.0

    @staticmethod
    @always_inline
    def extract_obs_and_masks[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        NUM_ACTIONS: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
        legal_masks: Tile[board_dtype, BATCH_SIZE, NUM_ACTIONS],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var player = Int(states[i, S_PLAYER])
        var my_off = 0 if player == 0 else 6
        var opp_off = 6 if player == 0 else 0

        # Planes 0-5: my pieces, 6-11: opp pieces (canonical coords)
        for pt in range(1, 7):
            var my_target = pt + my_off
            var opp_target = pt + opp_off
            for sq in range(64):
                var csq = sq if player == 0 else _flip_sq(sq)
                var p = Int(states[i, sq])
                if p == my_target:
                    obs[i, (pt - 1) * 64 + csq] = 1.0
                else:
                    obs[i, (pt - 1) * 64 + csq] = 0.0
                if p == opp_target:
                    obs[i, (pt + 5) * 64 + csq] = 1.0
                else:
                    obs[i, (pt + 5) * 64 + csq] = 0.0

        # Plane 12: castling
        var castling = Int(states[i, S_CASTLING])
        var has_c: Bool
        if player == 0:
            has_c = (castling & (CASTLE_WK | CASTLE_WQ)) != 0
        else:
            has_c = (castling & (CASTLE_BK | CASTLE_BQ)) != 0
        for sq in range(64):
            if has_c:
                obs[i, 12 * 64 + sq] = 1.0
            else:
                obs[i, 12 * 64 + sq] = 0.0

        # Plane 13: en passant
        var ep = Int(states[i, S_EP])
        for sq in range(64):
            var csq = sq if player == 0 else _flip_sq(sq)
            if ep >= 0 and csq == ep:
                obs[i, 13 * 64 + csq] = 1.0
            else:
                obs[i, 13 * 64 + csq] = 0.0

        # Legal mask
        _ = ChessEnv._gpu_gen_legal_mask_and_count[
            BATCH_SIZE, STATE_SIZE, NUM_ACTIONS
        ](states, legal_masks, i, player)

    # ========================================================================
    # GPU Launcher Methods (GPUTwoPlayerDiscreteEnv trait)
    # ========================================================================

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        actions_buf: DeviceBuffer[board_dtype],
        mut rewards_buf: DeviceBuffer[board_dtype],
        mut dones_buf: DeviceBuffer[board_dtype],
        mut terminated_buf: DeviceBuffer[board_dtype],
        mut obs_buf: DeviceBuffer[board_dtype],
        mut legal_masks_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64 = 0,
        rng_counter_ptr: Optional[Pointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var actions = tile[BATCH_SIZE](actions_buf)
        var rewards = tile[BATCH_SIZE](rewards_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        var terminated_out = tile[BATCH_SIZE](terminated_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)
        var legal_masks = tile[BATCH_SIZE, 4672](legal_masks_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def step_wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[board_dtype, BATCH_SIZE],
            rewards: Tile[board_dtype, BATCH_SIZE],
            dones: Tile[board_dtype, BATCH_SIZE],
            terminated_out: Tile[board_dtype, BATCH_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, 4672],
        ):
            ChessEnv.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones
            )
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]
            ChessEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 4672
            ](states, obs, legal_masks)

        ctx.enqueue_function[step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE]
        ):
            ChessEnv.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[wrapper](
            states, grid_dim=(BLOCKS,), block_dim=(Self.TPB,)
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        mut dones_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64,
        rng_counter_ptr: Optional[Pointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            dones: Tile[board_dtype, BATCH_SIZE],
        ):
            ChessEnv.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                states, dones
            )

        ctx.enqueue_function[wrapper](
            states, dones, grid_dim=(BLOCKS,), block_dim=(Self.TPB,)
        )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[board_dtype],
        mut obs_buf: DeviceBuffer[board_dtype],
        mut legal_masks_buf: DeviceBuffer[board_dtype],
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](rebind[Pointer[Scalar[board_dtype], MutAnyOrigin]]( states_buf.unsafe_ptr() ))
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)
        var legal_masks = tile[BATCH_SIZE, 4672](legal_masks_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, 4672],
        ):
            ChessEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 4672
            ](states, obs, legal_masks)

        ctx.enqueue_function[wrapper](
            states, obs, legal_masks, grid_dim=(BLOCKS,), block_dim=(Self.TPB,)
        )
