# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import tempfile
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import TorchscriptGreedyBatchedRNNTInfer
from nemo.utils import logging


"""
Script to compare the outputs of a NeMo Pytorch based RNNT Model and its Torchscript exported representation.

# Compare a NeMo and Torchscript model
python infer_transducer_ts.py \
    --nemo_model="<path to a .nemo file>" \
    OR
    --pretrained_model="<name of a pretrained model>" \
    --ts_encoder="<path to ts encoder file>" \
    --ts_decoder="<path to ts decoder-joint file>" \
    --ts_cfg="<path to a export ts model's config file>" \
    --dataset_manifest="<Either pass a manifest file path here>" \
    --audio_dir="<Or pass a directory containing preprocessed monochannel audio files>" \
    --max_symbold_per_step=5 \
    --batch_size=32 \
    --log
    
# Export and compare a NeMo and Torchscript model
python infer_transducer_ts.py \
    --nemo_model="<path to a .nemo file>" \
    OR
    --pretrained_model="<name of a pretrained model>" \
    --export \
    --dataset_manifest="<Either pass a manifest file path here>" \
    --audio_dir="<Or pass a directory containing preprocessed monochannel audio files>" \
    --max_symbold_per_step=5 \
    --batch_size=32 \
    --log

"""


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_model", type=str, default=None, required=False, help="Path to .nemo file",
    )
    parser.add_argument(
        '--pretrained_model', type=str, default=None, required=False, help='Name of a pretrained NeMo file'
    )
    parser.add_argument('--ts_encoder', type=str, default=None, required=False, help="Path to ts encoder model")
    parser.add_argument(
        '--ts_decoder', type=str, default=None, required=False, help="Path to ts decoder + joint model"
    )
    parser.add_argument(
        '--ts_cfg', type=str, default=None, required=False, help='Path to the yaml config of the exported model'
    )
    parser.add_argument('--threshold', type=float, default=0.01, required=False)

    parser.add_argument('--dataset_manifest', type=str, default=None, required=False, help='Path to dataset manifest')
    parser.add_argument('--audio_dir', type=str, default=None, required=False, help='Path to directory of audio files')
    parser.add_argument('--audio_type', type=str, default='wav', help='File format of audio')

    parser.add_argument(
        '--export', action='store_true', help="Whether to export the model into torchscript prior to eval"
    )
    parser.add_argument('--max_symbold_per_step', type=int, default=5, required=False, help='Number of decoding steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize')
    parser.add_argument('--log', action='store_true', help='Log the predictions between pytorch and torchscript')

    args = parser.parse_args()
    return args


def assert_args(args):
    if args.nemo_model is None and args.pretrained_model is None:
        raise ValueError(
            "`nemo_model` or `pretrained_model` must be passed ! It is required for decoding the RNNT tokens "
            "and ensuring predictions match between Torch and Torchscript."
        )

    if args.nemo_model is not None and args.pretrained_model is not None:
        raise ValueError(
            "`nemo_model` and `pretrained_model` cannot both be passed ! Only one can be passed to this script."
        )

    if args.ts_cfg is None:
        raise ValueError(
            "Must provide the yaml config of the exported model. You can obtain it by loading the "
            "nemo model and then using OmegaConf.save(model.cfg, 'cfg.yaml')"
        )

    if args.export and (args.ts_encoder is not None or args.ts_decoder is not None):
        raise ValueError("If `export` is set, then `ts_encoder` and `ts_decoder` arguments must be None")

    if args.audio_dir is None and args.dataset_manifest is None:
        raise ValueError("Both `dataset_manifest` and `audio_dir` cannot be None!")

    if args.audio_dir is not None and args.dataset_manifest is not None:
        raise ValueError("Submit either `dataset_manifest` or `audio_dir`.")

    if int(args.max_symbold_per_step) < 1:
        raise ValueError("`max_symbold_per_step` must be an integer > 0")


def export_model_if_required(args, nemo_model):
    if args.export:
        nemo_model.export(output="temp_rnnt.ts", check_trace=True)
        OmegaConf.save(nemo_model.cfg, "ts_cfg.yaml")

        args.ts_encoder = "encoder-temp_rnnt.ts"
        args.ts_decoder = "decoder_joint-temp_rnnt.ts"
        args.ts_cfg = "ts_cfg.yaml"


def resolve_audio_filepaths(args):
    # get audio filenames
    if args.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(args.audio_dir.audio_dir, f"*.{args.audio_type}")))
    else:
        # get filenames from manifest
        filepaths = []
        with open(args.dataset_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])

    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    return filepaths


def main():
    args = parse_arguments()

    device = get_current_device()

    # Instantiate pytorch model
    if args.nemo_model is not None:
        nemo_model = args.nemo_model
        nemo_model = ASRModel.restore_from(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    elif args.pretrained_model is not None:
        nemo_model = args.pretrained_model
        nemo_model = ASRModel.from_pretrained(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    else:
        raise ValueError("Please pass either `nemo_model` or `pretrained_model` !")

    if torch.cuda.is_available():
        nemo_model = nemo_model.to('cuda')

    export_model_if_required(args, nemo_model)

    # Instantiate RNNT Decoding loop
    encoder_model = args.ts_encoder
    decoder_model = args.ts_decoder
    ts_cfg = OmegaConf.load(args.ts_cfg)
    max_symbols_per_step = args.max_symbold_per_step
    decoding = TorchscriptGreedyBatchedRNNTInfer(encoder_model, decoder_model, ts_cfg, device, max_symbols_per_step)

    audio_filepath = resolve_audio_filepaths(args)

    # Evaluate Pytorch Model (CPU/GPU)
    actual_transcripts = nemo_model.transcribe(audio_filepath, batch_size=args.batch_size)[0]

    # Evaluate Torchscript model
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_filepath:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': audio_filepath, 'batch_size': args.batch_size, 'temp_dir': tmpdir}

        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0

        temporary_datalayer = nemo_model._setup_transcribe_dataloader(config)

        all_hypothesis = []
        for test_batch in tqdm(temporary_datalayer, desc="Torchscript Transcribing"):
            input_signal, input_signal_length = test_batch[0], test_batch[1]
            input_signal = input_signal.to(device)
            input_signal_length = input_signal_length.to(device)

            # Acoustic features
            processed_audio, processed_audio_len = nemo_model.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            # RNNT Decoding loop
            hypotheses = decoding(audio_signal=processed_audio, length=processed_audio_len)

            # Process hypothesis (map char/subword token ids to text)
            hypotheses = nemo_model.decoding.decode_hypothesis(hypotheses)  # type: List[str]

            # Extract text from the hypothesis
            texts = [h.text for h in hypotheses]

            all_hypothesis += texts
            del processed_audio, processed_audio_len
            del test_batch

    if args.log:
        for pt_transcript, ts_transcript in zip(actual_transcripts, all_hypothesis):
            print(f"Pytorch Transcripts        : {pt_transcript}")
            print(f"Torchscript Transcripts    : {ts_transcript}")
        print()

    # Measure error rate between torchscript and pytorch transcipts
    pt_ts_cer = word_error_rate(all_hypothesis, actual_transcripts, use_cer=True)
    assert pt_ts_cer < args.threshold, "Threshold violation !"

    print("Character error rate between Pytorch and Torchscript :", pt_ts_cer)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
