GT_FPS = 24
GT_RESOLUTION = (1280, 704)

MASK_KEY = "masks"
TRACKS_KEY = "tracks"
DEPTH_KEY = "depth"

SEGMENTATION_MODEL = "facebook/sam2.1-hiera-large"
DEPTH_MODEL = "depth-anything/Video-Depth-Anything-Large"
DEPTH_INPUT_SIZE = 384

N_QUERY_POINTS = 20
ERODE_ITERATIONS = 3

PROMPT_SUFFIX = (
    " Static shot, locked-off camera, fixed tripod, stationary framing."
    " No camera movement, no pan, no tilt, no zoom, no dolly, no drift, no handheld shake."
    " Real-time playback at natural speed, 24 fps, no slow motion, no time-lapse."
    " Realistic rigid-body physics: gravity, momentum, friction, and collisions consistent with reality."
    " Photorealistic, continuous single take, no cuts."
    " Motion begins on the first frame."
)
NEGATIVE_PROMPT = (
    "camera motion, camera shake, pan, tilt, zoom, dolly, parallax, handheld, tracking shot,"
    " slow motion, slo-mo, time-lapse, sped-up, motion blur,"
    " static start, frozen start, hesitation, delayed action, pause at beginning,"
    " floating object, levitation, hovering, teleportation, morphing, warping, deformed geometry,"
    " scene cut, jump cut, montage, transition, multiple shots,"
    " text, watermark, low quality, low resolution"
)
