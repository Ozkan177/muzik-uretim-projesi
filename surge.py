"""Drive Surge XT (free synth, VST3) from Python via pedalboard, loading its factory .fxp patches.
pedalboard has no patch browser, so we write the patch into the plugin's VST3 state: JUCE wraps the component state
as XML with its own base64 flavour; Surge's component state is the same 'sub3' patch chunk an .fxp file holds."""
import glob, os, re
from pedalboard import load_plugin

SURGE = os.path.abspath("tools/surge/Surge Synth Team/Surge XT.vst3/Contents/x86_64-win/Surge XT.vst3")
PATCHES = os.path.abspath("tools/surge/Surge Synth Team/SurgeXTData/patches_factory")
_T = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def juce_b64_decode(s):
    size, enc = s.split(".", 1)
    size = int(size)
    out = bytearray(size)
    pos = 0
    for ch in enc:
        v = _T.index(ch)
        for b in range(6):  # bits are packed little-endian, 6 at a time
            if pos >> 3 < size and v >> b & 1:
                out[pos >> 3] |= 1 << (pos & 7)
            pos += 1
    return bytes(out)


def juce_b64_encode(data):
    n_chars = (len(data) * 8 + 5) // 6
    out = []
    for i in range(n_chars):
        v = 0
        for b in range(6):
            pos = i * 6 + b
            if pos >> 3 < len(data) and data[pos >> 3] >> (pos & 7) & 1:
                v |= 1 << b
        out.append(_T[v])
    return f"{len(data)}." + "".join(out)


def patch(category, name):
    return os.path.join(PATCHES, category, name + ".fxp")


def load(fxp_path):
    """A Surge XT instance with the given factory patch loaded."""
    p = load_plugin(SURGE)
    state = p.raw_state
    xml = state[8:].decode("utf-8", "replace")
    comp = re.search(r"<IComponent>(.*?)</IComponent>", xml, re.S).group(1)
    blob = juce_b64_decode(comp)
    chunk = open(fxp_path, "rb").read()[60:]  # after the 60-byte FPCh header the Surge patch chunk begins ('sub3...')
    assert blob.startswith(b"sub3") and chunk.startswith(b"sub3"), "unexpected Surge state layout"
    body = xml.replace(comp, juce_b64_encode(chunk)).encode("utf-8")
    size = len(body) - (1 if body.endswith(bytes(1)) else 0)  # the size field excludes the trailing NUL
    p.raw_state = state[:4] + size.to_bytes(4, "little") + body
    p([], duration=0.2, sample_rate=44100)  # Surge applies a new patch on its next audio block
    return p


def list_patches(category):
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(PATCHES, category, "*.fxp")))


if __name__ == "__main__":
    p = load_plugin(SURGE)
    blob = juce_b64_decode(re.search(r"<IComponent>(.*?)</IComponent>", p.raw_state[8:].decode(), re.S).group(1))
    assert juce_b64_decode(juce_b64_encode(blob)) == blob  # codec round-trip
    print("codec ok")
