# +--------------------------------------------------------------------------+ #
# | A picture to a VLM — the camera-frame path
# +--------------------------------------------------------------------------+ #
"""Ask a vision-language model about an image.

    pixi run mojo run -I . examples/ai/vision_describe.mojo scene.png
    pixi run mojo run -I . examples/ai/vision_describe.mojo scene.png anthropic "Is the cube in the bowl?"

Any PNG works. From a live render or camera, `noeira.io.png.encode_png` turns
an RGB frame into the bytes this sends — no file needed.

⚠ SEND SMALL FRAMES. A 1280x720 PNG is ~1 MB of base64 on every turn of a
conversation; models see images at ~1 MP anyway and a 512 px frame is plenty
to ask "where is the cube". Resize first (`noeira.io.image`).
"""

from std.sys import argv

from noeira.ai.chat import ChatClient, Conversation
from noeira.io.fileio import read_file_bytes


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: vision_describe.mojo <image.png> [spec] [question]")
    var path = String(args[1])
    var spec = String(args[2]) if len(args) > 2 else String("hf")
    var question = (
        String(args[3]) if len(args) > 3
        else String("Describe this image in two sentences for a robot operator.")
    )
    var llm = ChatClient.from_spec(spec)
    llm.max_tokens = 512
    if spec.startswith("hf"):
        llm.extra("chat_template_kwargs", '{"enable_thinking": false}')
    var png = read_file_bytes(path)
    var media = String("image/jpeg") if path.endswith(".jpg") or path.endswith(".jpeg") else String("image/png")
    var conv = Conversation()
    conv.user_image(question, png, media)
    var r = conv.send(llm)
    print(r.text)
    print("--", r.model, "|", r.input_tokens, "in |", r.latency_ms, "ms")
