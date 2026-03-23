#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from gui_data.constants import (
    DEFAULT,
    DEFAULT_DATA,
    DEMUCS_ARCH_TYPE,
    DEF_OPT,
    INST_STEM,
    MDX_ARCH_TYPE,
    NO_MODEL,
    VR_ARCH_TYPE,
    VR_ARCH_PM,
    WAV,
)
from separate import SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR, clear_gpu_cache
import UVR


PROCESS_METHOD_MAP = {
    "vr": VR_ARCH_TYPE,
    "mdx": MDX_ARCH_TYPE,
    "demucs": DEMUCS_ARCH_TYPE,
}


class SimpleVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class CliRoot:
    def __init__(self, args):
        defaults = dict(DEFAULT_DATA)
        process_method = PROCESS_METHOD_MAP[args.process_method]
        defaults.update(
            {
                "chosen_process_method": process_method,
                "vr_model": args.model if process_method == VR_ARCH_TYPE else defaults["vr_model"],
                "mdx_net_model": args.model if process_method == MDX_ARCH_TYPE else defaults["mdx_net_model"],
                "demucs_model": args.model if process_method == DEMUCS_ARCH_TYPE else defaults["demucs_model"],
                "aggression_setting": args.aggression,
                "window_size": args.window_size,
                "is_gpu_conversion": args.gpu,
                "save_format": WAV,
                "wav_type_set": "PCM_16",
                "is_primary_stem_only": False,
                "is_secondary_stem_only": False,
                "export_path": str(args.output_dir),
                "input_paths": [str(path) for path in args.inputs],
            }
        )

        self.vr_hash_MAPPER = UVR.load_model_hash_data(UVR.VR_HASH_JSON)
        self.mdx_hash_MAPPER = UVR.load_model_hash_data(UVR.MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = UVR.load_model_hash_data(UVR.MDX_MODEL_NAME_SELECT)
        self.demucs_name_select_MAPPER = UVR.load_model_hash_data(UVR.DEMUCS_MODEL_NAME_SELECT)

        self.device_set_var = SimpleVar(DEFAULT)
        self.is_gpu_conversion_var = SimpleVar(defaults["is_gpu_conversion"])
        self.is_normalization_var = SimpleVar(defaults["is_normalization"])
        self.is_primary_stem_only_var = SimpleVar(defaults["is_primary_stem_only"])
        self.is_secondary_stem_only_var = SimpleVar(defaults["is_secondary_stem_only"])
        self.denoise_option_var = SimpleVar(defaults["denoise_option"])
        self.is_mdx_c_seg_def_var = SimpleVar(defaults["is_mdx_c_seg_def"])
        self.is_mdx23_combine_stems_var = SimpleVar(defaults["is_mdx23_combine_stems"])
        self.mdx_batch_size_var = SimpleVar(defaults["mdx_batch_size"])
        self.mdxnet_stems_var = SimpleVar(defaults["mdx_stems"])
        self.overlap_var = SimpleVar(defaults["overlap"])
        self.overlap_mdx_var = SimpleVar(defaults["overlap_mdx"])
        self.overlap_mdx23_var = SimpleVar(defaults["overlap_mdx23"])
        self.semitone_shift_var = SimpleVar(defaults["semitone_shift"])
        self.wav_type_set_var = SimpleVar(defaults["wav_type_set"])
        self.mp3_bit_set_var = SimpleVar(defaults["mp3_bit_set"])
        self.save_format_var = SimpleVar(defaults["save_format"])
        self.is_invert_spec_var = SimpleVar(defaults["is_invert_spec"])
        self.demucs_stems_var = SimpleVar(defaults["demucs_stems"])
        self.is_demucs_combine_stems_var = SimpleVar(defaults["is_demucs_combine_stems"])
        self.mdx_segment_size_var = SimpleVar(defaults["mdx_segment_size"])
        self.margin_var = SimpleVar(defaults["margin"])
        self.compensate_var = SimpleVar(defaults["compensate"])
        self.is_demucs_pre_proc_model_activate_var = SimpleVar(defaults["is_demucs_pre_proc_model_activate"])
        self.is_demucs_pre_proc_model_inst_mix_var = SimpleVar(defaults["is_demucs_pre_proc_model_inst_mix"])
        self.demucs_is_secondary_model_activate_var = SimpleVar(defaults["demucs_is_secondary_model_activate"])
        self.mdx_is_secondary_model_activate_var = SimpleVar(defaults["mdx_is_secondary_model_activate"])
        self.margin_demucs_var = SimpleVar(defaults["margin_demucs"])
        self.shifts_var = SimpleVar(defaults["shifts"])
        self.is_split_mode_var = SimpleVar(defaults["is_split_mode"])
        self.segment_var = SimpleVar(defaults["segment"])
        self.is_chunk_demucs_var = SimpleVar(defaults["is_chunk_demucs"])
        self.demucs_model_var = SimpleVar(defaults["demucs_model"])
        self.mdx_net_model_var = SimpleVar(defaults["mdx_net_model"])
        self.is_save_inst_set_vocal_splitter_var = SimpleVar(False)
        self.is_deverb_vocals_var = SimpleVar(defaults["is_deverb_vocals"])
        self.deverb_vocal_opt_var = SimpleVar(defaults["deverb_vocal_opt"])
        self.is_match_frequency_pitch_var = SimpleVar(defaults["is_match_frequency_pitch"])
        self.vr_is_secondary_model_activate_var = SimpleVar(False)
        self.aggression_setting_var = SimpleVar(defaults["aggression_setting"])
        self.is_tta_var = SimpleVar(defaults["is_tta"])
        self.is_post_process_var = SimpleVar(defaults["is_post_process"])
        self.window_size_var = SimpleVar(defaults["window_size"])
        self.batch_size_var = SimpleVar(DEF_OPT)
        self.crop_size_var = SimpleVar(defaults["crop_size"])
        self.is_high_end_process_var = SimpleVar(defaults["is_high_end_process"])
        self.post_process_threshold_var = SimpleVar(defaults["post_process_threshold"])
        self.export_path_var = SimpleVar(defaults["export_path"])
        self.vr_model_var = SimpleVar(defaults["vr_model"])
        self.wav_type_set = defaults["wav_type_set"]

        self.vr_voc_inst_secondary_model_var = SimpleVar(NO_MODEL)
        self.vr_other_secondary_model_var = SimpleVar(NO_MODEL)
        self.vr_bass_secondary_model_var = SimpleVar(NO_MODEL)
        self.vr_drums_secondary_model_var = SimpleVar(NO_MODEL)
        self.vr_voc_inst_secondary_model_scale_var = SimpleVar(defaults["vr_voc_inst_secondary_model_scale"])
        self.vr_other_secondary_model_scale_var = SimpleVar(defaults["vr_other_secondary_model_scale"])
        self.vr_bass_secondary_model_scale_var = SimpleVar(defaults["vr_bass_secondary_model_scale"])
        self.vr_drums_secondary_model_scale_var = SimpleVar(defaults["vr_drums_secondary_model_scale"])

    def check_only_selection_stem(self, _checktype):
        return False

    def process_determine_vocal_split_model(self):
        return None

    def process_determine_secondary_model(self, _process_method, _main_model_primary_stem, is_primary_stem_only=False, is_secondary_stem_only=False):
        return (None, None)


class CliRunner:
    def __init__(self):
        self.cached_sources = {
            VR_ARCH_TYPE: {},
            MDX_ARCH_TYPE: {},
            DEMUCS_ARCH_TYPE: {},
        }

    def set_progress_bar(self, step, inference_iterations=0):
        progress = max(0.0, min(1.0, step + inference_iterations))
        print(f"[progress] {progress:.0%}", file=sys.stderr)

    def write_to_console(self, message, base_text=""):
        text = f"{base_text}{message}".rstrip()
        if text:
            print(text, file=sys.stderr)

    def process_iteration(self):
        return None

    def cached_source_callback(self, process_method, model_name=None):
        source = self.cached_sources.get(process_method, {})
        return model_name, source.get(model_name)

    def cached_model_source_holder(self, process_method, secondary_sources, model_name):
        self.cached_sources.setdefault(process_method, {})[model_name] = secondary_sources

    def build_process_data(self, audio_file, export_path):
        audio_file_base = Path(audio_file).stem
        return {
            "model_data": None,
            "export_path": export_path,
            "audio_file_base": audio_file_base,
            "audio_file": audio_file,
            "set_progress_bar": self.set_progress_bar,
            "write_to_console": self.write_to_console,
            "process_iteration": self.process_iteration,
            "cached_source_callback": self.cached_source_callback,
            "cached_model_source_holder": self.cached_model_source_holder,
            "list_all_models": [],
            "is_ensemble_master": False,
            "is_4_stem_ensemble": False,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless CLI for UVR model inference."
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Input audio file(s).")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("cli_outputs"),
        help="Directory for separated stems.",
    )
    parser.add_argument(
        "--process-method",
        choices=tuple(PROCESS_METHOD_MAP.keys()),
        default="vr",
        help=f'Backend to use. "vr" maps to the GUI label "{VR_ARCH_PM}".',
    )
    parser.add_argument(
        "--model",
        default="5_HP-Karaoke-UVR",
        help="Model name. For VR models, use the basename without .pth.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="VR window size.",
    )
    parser.add_argument(
        "--aggression",
        type=int,
        default=5,
        help="VR aggression setting, matching the GUI value.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU conversion when available.",
    )
    parser.add_argument(
        "--save-both",
        action="store_true",
        help="Save both stems instead of instrumental only.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read input file paths from stdin, one path per line.",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args()


def configure_stem_saving(model_data, save_both):
    if save_both:
        model_data.is_primary_stem_only = False
        model_data.is_secondary_stem_only = False
        return

    if model_data.primary_stem == INST_STEM:
        model_data.is_primary_stem_only = True
        model_data.is_secondary_stem_only = False
    else:
        model_data.is_primary_stem_only = False
        model_data.is_secondary_stem_only = True


def validate_args(args):
    if not args.stdin and not args.inputs:
        raise SystemExit("Provide at least one input path or use --stdin.")

    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)


def build_model_data(args):
    process_method = PROCESS_METHOD_MAP[args.process_method]
    return UVR.ModelData(args.model, process_method)


def build_separator(model_data, process_data):
    if model_data.process_method == VR_ARCH_TYPE:
        return SeperateVR(model_data, process_data)
    if model_data.process_method == MDX_ARCH_TYPE:
        return SeperateMDXC(model_data, process_data) if model_data.is_mdx_c else SeperateMDX(model_data, process_data)
    if model_data.process_method == DEMUCS_ARCH_TYPE:
        return SeperateDemucs(model_data, process_data)
    raise SystemExit(f"Unsupported process method: {model_data.process_method}")


def expected_model_location(args):
    process_method = PROCESS_METHOD_MAP[args.process_method]
    if process_method == VR_ARCH_TYPE:
        return Path(UVR.VR_MODELS_DIR) / f"{args.model}.pth"
    if process_method == MDX_ARCH_TYPE:
        return Path(UVR.MDX_MODELS_DIR)
    if process_method == DEMUCS_ARCH_TYPE:
        return Path(UVR.DEMUCS_MODELS_DIR)
    return Path(".")


def result_output_paths(model_data, input_path, output_dir):
    audio_file_base = Path(input_path).stem
    output_paths = []

    if not model_data.is_secondary_stem_only:
        output_paths.append(str(output_dir / f"{audio_file_base}_({model_data.primary_stem}).wav"))

    if not model_data.is_primary_stem_only:
        output_paths.append(str(output_dir / f"{audio_file_base}_({model_data.secondary_stem}).wav"))

    return output_paths


def emit_result(input_path, ok, outputs=None, error=None):
    payload = {
        "input": str(input_path),
        "ok": ok,
    }

    if outputs is not None:
        payload["outputs"] = outputs

    if error is not None:
        payload["error"] = error

    print(json.dumps(payload), flush=True)


def iter_stdin_inputs():
    for raw_line in sys.stdin:
        yield raw_line.rstrip("\r\n")


def process_input(input_path, args, runner, model_data):
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")

    process_data = runner.build_process_data(str(path), str(args.output_dir))
    process_data["model_data"] = model_data
    process_data["list_all_models"] = [model_data.model_basename]
    print(f"[input] {path}", file=sys.stderr)
    separator = build_separator(model_data, process_data)
    separator.seperate()

    return result_output_paths(model_data, path, args.output_dir)


def main():
    args = parse_args()
    validate_args(args)

    cli_root = CliRoot(args)
    UVR.root = cli_root

    runner = CliRunner()
    model_data = build_model_data(args)

    if not model_data.model_status:
        raise SystemExit(
            f'Unable to load model "{args.model}" for process method "{args.process_method}". '
            f"Checked under {expected_model_location(args)}"
        )

    configure_stem_saving(model_data, args.save_both)
    input_iterable = iter_stdin_inputs() if args.stdin else (str(path) for path in args.inputs)

    for raw_input in input_iterable:
        if args.stdin and raw_input == "":
            error = "Empty input line."
            print(f"[error] {error}", file=sys.stderr)
            emit_result(raw_input, ok=False, error=error)
            continue

        try:
            outputs = process_input(raw_input, args, runner, model_data)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            emit_result(raw_input, ok=False, error=str(exc))
        else:
            emit_result(raw_input, ok=True, outputs=outputs)
        finally:
            clear_gpu_cache()


if __name__ == "__main__":
    main()
