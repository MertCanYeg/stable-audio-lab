"""Model registry and single source of truth for Stable Audio Lab models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    display_name: str
    parameters: str
    approx_size: str
    default_duration: float
    max_duration: float
    default_prompt: str
    description: str
    note: str = ""
    files: Tuple[Tuple[str, float], ...] = (
        ("model_config.json", 0.0),
        ("model.safetensors", 50.0),
        ("t5gemma-b-b-ul2/model.safetensors", 10.0),
    )
    examples: Union[List[List[Any]], Dict[str, List[List[Any]]]] = field(default_factory=list)


MODELS: Dict[str, ModelSpec] = {
    "small-music": ModelSpec(
        name="small-music",
        repo_id="stabilityai/stable-audio-3-small-music",
        display_name="🎵 Music (Small-Music)",
        parameters="433M",
        approx_size="~1.5 GB (~3.2 GB with T5)",
        default_duration=15.0,
        max_duration=120.0,
        default_prompt="Upbeat funky bassline with warm rhodes piano and crisp drums",
        description="Fast, lightweight music composition model trained for full stereo musical tracks.",
        note="💡 **Music Model:** Fast generation (~2s for 15s audio on modern GPUs). Maximum duration: 120s.",
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
        display_name="🔊 Sound Effects (Small-SFX)",
        parameters="433M",
        approx_size="~1.5 GB (~3.2 GB with T5)",
        default_duration=5.0,
        max_duration=120.0,
        default_prompt="TrackType: SFX, a funny high-pitched rubber clown nose squeak honk sound with a quick double squeeze",
        description="Specialized sound effects, foley, and environmental soundscape generator.",
        note="💡 **SFX Model:** Optimized for Foley, impacts, and environmental soundscapes. Start prompts with `TrackType: SFX` for best results.",
        examples=[
            ["TrackType: SFX, a funny high-pitched rubber clown nose squeak honk sound with a quick double squeeze", 3.0],
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
        display_name="🎛️ Medium (1.4B Quality)",
        parameters="1.4B",
        approx_size="~8.6 GB (~9.7 GB with T5)",
        default_duration=15.0,
        max_duration=380.0,
        default_prompt="An epic cinematic orchestral trailer theme with thundering percussion, brass swells, and soaring strings",
        description="Unified 1.4B flagship model capable of full production music and blockbuster sound effects up to 380 seconds.",
        note="💡 **Flagship Model:** Stable Audio 3 Medium (1.4B) trained up to **380s (~6.3 mins)**. High fidelity for both music and cinematic sound design.",
        examples={
            "🎵 Music & Cinematic": [
                ["Massive cinematic sci-fi orchestral trailer theme with thundering timpani, colossal brass swells, and soaring strings", 30.0],
                ["Ancient Nordic Viking folk music with resonant tagelharpa, bowed lyre, hypnotic shamanic frame drum, and deep vocal drone", 35.0],
                ["80s retro synthwave outrun anthem with driving analog arpeggios, punchy gated reverb snare, and soaring guitar lead", 30.0],
                ["Dark cyberpunk neo-noir soundtrack with solitary melancholic muted trumpet, deep sub-bass drone, and rainy neon city pads", 30.0],
                ["Soulful 70s Motown funk with live brass section, warm Hammond B3 organ, rhythmic wah-wah guitar, and melodic bass", 30.0],
                ["Epic fantasy highland soundtrack with evocative Celtic uilleann pipes, tin whistle, sweeping orchestral strings, and bodhrán", 35.0],
                ["Melodic organic deep house with subtle marimba plucks, smooth round sub-bass, crisp shakers, and lush sunset beach reverb", 30.0],
                ["Late night smoky noir jazz ballad with expressive tenor saxophone, brushed snare, and warm upright double bass", 30.0],
                ["Deep space ambient meditation soundscape with evolving granular shimmer pads, zero-gravity drone, and ethereal harmonic resonances", 45.0],
                ["Alternative 2000s post-grunge hard rock anthem with wall-of-sound distorted guitars, punchy arena drums, and soaring melody", 30.0],
            ],
            "🔊 Sound Effects & Foley": [
                ["TrackType: SFX, colossal cinematic explosion with deep sub-bass shockwave, flying debris, and reverberant metallic tail", 6.0],
                ["TrackType: SFX, thunderstorm inside a dense tropical rainforest with raindrops hitting large leaves and distant rolling thunder", 30.0],
                ["TrackType: SFX, massive robotic mech powering up with mechanical servo whines, hydraulic hiss, and heavy metallic footsteps", 8.0],
                ["TrackType: SFX, ominous mythical monster roar echoing inside a cavernous underground cave with terrifying guttural growl", 6.0],
                ["TrackType: SFX, futuristic hovercar soaring past at high speed with a Doppler pitch shift and turbo jet exhaust whine", 5.0],
                ["TrackType: SFX, medieval castle siege with flaming catapult boulders launching, wooden wheels creaking, and ambient battle chaos", 15.0],
            ],
        },
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """Retrieve ModelSpec by name, raising a clear ValueError if not registered."""
    if model_name not in MODELS:
        valid = list(MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Valid choices: {valid}")
    return MODELS[model_name]
