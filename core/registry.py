"""Model specifications, durations, and curated prompt examples for Stable Audio Lab."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ModelNotFoundError


@dataclass(frozen=True)
class ModelSpec:
    """Specification and curated metadata for a Stable Audio 3 model variant."""

    name: str
    repo_id: str
    default_prompt: str
    default_duration: float
    max_duration: float
    description: str = ""
    examples: list[list[str | float]] = field(default_factory=list)


MODELS: dict[str, ModelSpec] = {
    "small-music": ModelSpec(
        name="small-music",
        repo_id="stabilityai/stable-audio-3-small-music",
        default_prompt="Upbeat funky bassline with warm rhodes piano and crisp drums",
        default_duration=15.0,
        max_duration=120.0,
        description="Stereo music generation (up to 120s).",
        examples=[
            ["Upbeat funky bassline with warm rhodes piano and crisp drums", 15.0],
            ["2000s alternative rock with heavy drop-tuned guitars, driving drums, and angsty melodic chorus", 30.0],
            ["70s Anatolian psychedelic rock with electric bağlama fuzz lead, groovy bassline, and swirling phaser guitars", 30.0],
            ["Rainy Tokyo lo-fi jazzhop beat with gentle rain, dusty vinyl crackle, warm Rhodes piano, and boom-bap swing", 30.0],
            ["80s Japanese City Pop with bright slap bass, funk rhythm guitar, electric piano, and sparkling brass stabs", 30.0],
            ["Smooth jazz trumpet solo over mellow acoustic drums and walking upright bass in a late night club", 20.0],
            ["Traditional Turkish classical art music with melancholic Oud, delicate Kanun flourishes, and soft Bendir", 25.0],
            ["Passionate Spanish flamenco with rapid nylon guitar rasgueado, wooden cajon, and rhythmic palmas", 20.0],
            ["Catchy 90s French touch nu-disco with filtered slap bass, four-on-the-floor kick, and joyful phaser synths", 25.0],
            ["Raw acoustic Delta blues with bottleneck slide resonator guitar and rhythmic wooden floor stomps", 20.0],
            ["Cozy 16-bit SNES RPG town theme with playful marimba, gentle wooden flute, and warm chiptune bass", 20.0],
            ["Desert stoner rock with heavy fuzz down-tuned guitars, thick bass, and swinging dry room drums", 25.0],
        ],
    ),
    "small-sfx": ModelSpec(
        name="small-sfx",
        repo_id="stabilityai/stable-audio-3-small-sfx",
        default_prompt="TrackType: SFX, funny rubber clown nose squeak honk sound with double squeeze",
        default_duration=5.0,
        max_duration=120.0,
        description="Sound effects and foley generation (up to 120s).",
        examples=[
            ["TrackType: SFX, funny rubber clown nose squeak honk sound with double squeeze", 3.0],
            ["TrackType: SFX, deep campfire crackling and popping in a dense pine forest with gentle whistling night wind", 15.0],
            ["TrackType: SFX, powerful sci-fi plasma rifle blaster shot with metallic electrical dissipation", 3.0],
            ["TrackType: SFX, heavy thunderstorm with torrential rain pouring against a window and distant rolling thunder", 30.0],
            ["TrackType: SFX, classic cartoon boing spring bounce sound effect, comedic and bouncy", 3.0],
            ["TrackType: SFX, heavy pneumatic spaceship airlock door depressurizing with a loud industrial hiss", 5.0],
            ["TrackType: SFX, futuristic laser sword igniting with a sharp hum and vibrating idle buzz", 4.0],
            ["TrackType: SFX, gentle ocean waves lapping against a pebbly beach with distant seagulls", 20.0],
            ["TrackType: SFX, wooden door creaking slowly open in an eerie hallway followed by a heavy latch click", 5.0],
            ["TrackType: SFX, vintage typewriter rapidly clacking with a carriage return bell ding", 6.0],
            ["TrackType: SFX, bustling cozy coffee shop ambience with quiet chatter, clinking espresso cups, and background murmur", 20.0],
            ["TrackType: SFX, deep cinematic impact sub-bass braam hit with long reverberant decay", 6.0],
        ],
    ),
    "medium": ModelSpec(
        name="medium",
        repo_id="stabilityai/stable-audio-3-medium",
        default_prompt="An epic cinematic orchestral trailer theme with thundering percussion, brass swells, and soaring strings",
        default_duration=15.0,
        max_duration=380.0,
        description="Flagship quality model for music and sound design (up to 380s).",
        examples=[
            ["Massive cinematic sci-fi orchestral trailer theme with thundering timpani, colossal brass swells, and soaring strings", 30.0],
            ["Ancient Nordic Viking folk music with resonant tagelharpa, bowed lyre, hypnotic frame drum, and vocal drone", 35.0],
            ["80s retro synthwave outrun anthem with driving analog arpeggios, punchy gated reverb snare, and soaring guitar lead", 30.0],
            ["Dark cyberpunk neo-noir soundtrack with solitary melancholic muted trumpet and deep sub-bass drone", 30.0],
            ["Soulful 70s Motown funk with live brass section, warm Hammond B3 organ, and melodic bass", 30.0],
            ["Epic fantasy highland soundtrack with Celtic uilleann pipes, tin whistle, sweeping orchestral strings, and bodhran", 35.0],
            ["Melodic organic deep house with subtle marimba plucks, smooth round sub-bass, crisp shakers, and sunset beach reverb", 30.0],
            ["Late night smoky noir jazz ballad with expressive tenor saxophone, brushed snare, and upright double bass", 30.0],
            ["Deep space ambient meditation soundscape with evolving granular shimmer pads, zero-gravity drone, and harmonic resonances", 45.0],
            ["Alternative 2000s post-grunge hard rock anthem with wall-of-sound distorted guitars, arena drums, and soaring melody", 30.0],
            ["TrackType: SFX, colossal cinematic explosion with deep sub-bass shockwave and reverberant tail", 6.0],
            ["TrackType: SFX, thunderstorm inside a dense tropical rainforest with raindrops hitting large leaves and distant thunder", 30.0],
            ["TrackType: SFX, massive robotic mech powering up with mechanical servo whines and hydraulic hiss", 8.0],
            ["TrackType: SFX, ominous mythical monster roar echoing inside a cavernous cave with terrifying guttural growl", 6.0],
            ["TrackType: SFX, futuristic hovercar soaring past at high speed with Doppler pitch shift and turbo jet exhaust whine", 5.0],
            ["TrackType: SFX, medieval castle siege with flaming catapult boulders launching, wooden wheels creaking, and battle ambience", 15.0],
        ],
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """Retrieve ModelSpec by name, raising ModelNotFoundError if unrecognized."""
    if model_name not in MODELS:
        valid = ", ".join(repr(k) for k in MODELS.keys())
        raise ModelNotFoundError(f"Unknown model '{model_name}'. Valid choices: {valid}")
    return MODELS[model_name]
