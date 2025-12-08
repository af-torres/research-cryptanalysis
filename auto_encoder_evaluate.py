import torch

import argparse
import random
import string

from model import build_auto_encoder as build_model
from datasets import load_dataset
from utils import load_results, get_accuracy, plot_acc_line

if torch.cuda.is_available():
    dev = "cuda"
    print('Running on CUDA')
else: 
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)

alphabet = string.ascii_lowercase + " "

parser = argparse.ArgumentParser(
    description="Script that evaluates and creates report for a specified trained model."
)
parser.add_argument(
    "-m",
    "--model_version",
    type=str,
    choices=[
        "one_hot",
        "simple",
        "embedding",
        "reduced_embedding",
        "reduced_by_char_embedding",
        "by_char_embedding",
        "lstm"
    ],
    required=True,
    help="Specify the model type. Must be one of: 'reduced' or 'simple'."
)
parser.add_argument(
    "-id",
    "--train_id",
    type=str,
    required=True,
    help="Specify the model's training ID."
)
parser.add_argument(
    "--input_noise",
    type=float,
    default=0,
    help="Specify the percentage of noise perturbation to add to inputs (0-1)."
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    choices=[
        "eng_sentences",
        "random"
    ],
    required=True
)
args = parser.parse_args()

model_version = args.model_version
train_id = args.train_id
input_noise_p = args.input_noise
dataset = args.dataset

datasets_map = dict(
    eng_sentences = [
        {
            "DATASET_ENC": './data/eng_sentences/substitutionCipherArr-encryptedEngSeq.csv',
            "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
            "RESULTS_DIR": "./results/eng_sentences/",
            "NAME": "eng_sentences-substitution",
        },
        {
            "DATASET_ENC": './data/eng_sentences/transpositionCipherArr-encryptedEngSeq.csv',
            "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
            "RESULTS_DIR": "./results/eng_sentences/",
            "NAME": "eng_sentences-transposition",
        },
        {
            "DATASET_ENC": './data/eng_sentences/productCipherArr-encryptedEngSeq.csv',
            "DATASET_ORI": './data/eng_sentences/arr-decryptedEngSeq.csv',
            "RESULTS_DIR": "./results/eng_sentences/",
            "NAME": "eng_sentences-product",
        },
    ],
    random = [
        {
            "DATASET_ENC": './data/random/substitutionCipherArr-encryptedRandomAsciiCharSeq.csv',
            "DATASET_ORI": './data/random/arr-decryptedRandomAsciiCharSeq.csv',
            "RESULTS_DIR": "./results/random/",
            "NAME": "random-substitution",
        },
        {
            "DATASET_ENC": './data/random/transpositionCipherArr-encryptedRandomAsciiCharSeq.csv',
            "DATASET_ORI": './data/random/arr-decryptedRandomAsciiCharSeq.csv',
            "RESULTS_DIR": "./results/random/",
            "NAME": "random-transposition",
        },
        {
            "DATASET_ENC": './data/random/productCipherArr-encryptedRandomAsciiCharSeq.csv',
            "DATASET_ORI": './data/random/arr-decryptedRandomAsciiCharSeq.csv',
            "RESULTS_DIR": "./results/random/",
            "NAME": "random-product",
        },
    ]
)

DATASETS = datasets_map.get(dataset)
assert DATASETS

aggregated_summary = f"{train_id}: model={model_version}; input_noise={input_noise_p}"

for i, d in enumerate(DATASETS):
    torch.manual_seed(1234)
    random.seed(1234)

    train, validation, test = load_dataset(d.get("DATASET_ORI"), d.get("DATASET_ENC"), device, input_noise_p)
    X_tr, Y_tr = train
    X_vl, Y_vl = validation
    X_ts, Y_ts = test

    # num_classes = len(torch.unique(X_tr))
    num_classes = len(alphabet) + 1

    seq_len = X_tr.shape[1]
    
    results_dir = d.get("RESULTS_DIR", "")
    dataset_name = d.get("NAME")
    results_file = results_dir + f"{model_version}/"  + dataset_name + "-autoencoder-" + train_id + ".pkl" # type: ignore
    results = load_results(results_file)
    hyper_params = results.get("hyper-params")
    assert hyper_params is not None
    weights = results.get("weights", None)
    assert weights is not None

    build = build_model(
        model_version,
        num_classes, seq_len,
        **hyper_params
    )
    m = build.get("model")

    m.load_state_dict(weights) # type: ignore
    m.to(device) # type: ignore
     
    tr_row_acc = get_accuracy(m, X_tr, Y_tr, axis=1, sort=True)
    tr_col_acc = get_accuracy(m, X_tr, Y_tr, axis=0)
    plot_acc_line(
        1-tr_row_acc, 1-tr_col_acc,
        output_filename=f"results/pictures/auto_encoder_{model_version}-tr_{dataset_name}-{train_id}",
        subtitle="Row and Column Miss-classification", metric_type="Miss-classification"
    )
    
    vl_row_acc = get_accuracy(m, X_vl, Y_vl, axis=1, sort=True)
    vl_col_acc = get_accuracy(m, X_vl, Y_vl, axis=0)      
    plot_acc_line(
        1-vl_row_acc, 1-vl_col_acc,
        output_filename=f"results/pictures/auto_encoder_{model_version}-vl_{dataset_name}-{train_id}",
        subtitle="Row and Column Miss-classification", metric_type="Miss-classification"
    )
    
    ts_row_acc = get_accuracy(m, X_ts, Y_ts, axis=1, sort=True)
    ts_col_acc = get_accuracy(m, X_ts, Y_ts, axis=0)
    plot_acc_line(
        1-ts_row_acc, 1-ts_col_acc, 
        output_filename=f"results/pictures/auto_encoder_{model_version}-ts_{dataset_name}-{train_id}", 
        subtitle="Row and Column Miss-classification", metric_type="Miss-classification"
    )

    tr_row_acc = get_accuracy(m, X_tr, Y_tr)
    vl_row_acc = get_accuracy(m, X_vl, Y_vl)
    ts_row_acc = get_accuracy(m, X_ts, Y_ts)
    
    aggregated_summary += f"; {dataset_name}-tr-acc={tr_row_acc}; {dataset_name}-vl-acc={vl_row_acc}; {dataset_name}-ts-acc={ts_row_acc}"
    
    with open(f"results/reports/auto_encoder-{dataset_name}-{train_id}.md", "w") as report_file:        
        report_line = f"Accuracy acchieved:\ntraining: {tr_row_acc}\nvalidation: {vl_row_acc}\ntesting: {ts_row_acc}"
        report_line += "\n#####################################\n"
        report_file.write(report_line)

with open("report_log", "a") as file:
    file.write(f"{aggregated_summary}\n")
