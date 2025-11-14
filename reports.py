import subprocess
import sys

EVALUATOR_SCRIPT="./auto_encoder_evaluate.py"
TRAINING_LOG_FILE="./training_log"

def run_evaluation(model, train_id, input_noise):
    interpreter_path = sys.executable
    result = subprocess.run([
        interpreter_path, EVALUATOR_SCRIPT, 
        f"--model_version={model}",
        f"--train_id={train_id}",
        f"--input_noise={input_noise}"
    ], capture_output=True, text=True)

    print("Output:", result.stdout)

def read_training_log():
    with open(TRAINING_LOG_FILE, 'r', encoding='utf-8') as file:
        log_data = file.read()

    execution_logs = []
    for execution in log_data.splitlines():
        config_map = dict()
        
        train_id, config = execution.split(":")
        config_map["train_id"] = train_id

        config_vars = config.split(";")
        for var in config_vars:
            var = var.strip()
            k, v = var.split("=")
            config_map[k] = v

        execution_logs.append(config_map)

    return execution_logs

if __name__ == "__main__":
    execution_logs = read_training_log()
    for e in execution_logs:
        run_evaluation(**e)

