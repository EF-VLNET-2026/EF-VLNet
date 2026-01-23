from transformers import AutoProcessor, AutoModelForVision2Seq
import os
from pathlib import Path
import torch
import psutil
import time

def check_system_resources():
    print("Checking system resources...")

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    print(f"Available memory: {available_gb:.1f} GB")

    if available_gb < 8:
        print("Warning: Available memory is less than 8GB, download might fail")
        print("Suggestion: Close other programs to free up memory")

    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    print(f"Available disk space: {free_gb:.1f} GB")

    if free_gb < 20:
        print("Warning: Available disk space is less than 20GB")
        return False

    return True


def download_lingshu_model_optimized():
    model_name = "lingshu-medical-mllm/Lingshu-7B"
    local_dir = "/path/to/your/own/models/Lingshu-7B"

    print("=" * 60)
    print("Lingshu-7B Model Downloader (Improved)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Save path: {os.path.abspath(local_dir)}")
    print()

    if not check_system_resources():
        print("\nInsufficient system resources, please clean up and try again")
        return None, None

    os.makedirs(local_dir, exist_ok=True)

    try:
        print("\n" + "=" * 60)
        print("Step 1/2: Downloading Processor")
        print("=" * 60)

        processor_path = Path(local_dir) / "preprocessor_config.json"
        if processor_path.exists():
            print("Processor already exists, skipping download")
            processor = AutoProcessor.from_pretrained(
                local_dir,
                trust_remote_code=True
            )
        else:
            print("Starting processor download...")
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                resume_download=True
            )
            processor.save_pretrained(local_dir)
            print("Processor download complete")

        print("\n" + "=" * 60)
        print("Step 2/2: Downloading Model Weights")
        print("=" * 60)
        print("Tip: This step may take 15-30 minutes")
        print("Model size is approx 14GB, containing 4 split files")
        print()

        existing_files = list(Path(local_dir).glob("*.safetensors")) + \
                        list(Path(local_dir).glob("*.bin"))
        if existing_files:
            print(f"Found {len(existing_files)} existing files, resuming download...")

        print("Starting model download...")
        print("(If network interrupts, run script again to resume)")
        print()

        start_time = time.time()

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            resume_download=True
        )

        elapsed_time = time.time() - start_time
        print(f"\nModel download complete (Time: {elapsed_time/60:.1f} minutes)")

        print("\nSaving model locally...")
        model.save_pretrained(
            local_dir,
            safe_serialization=True
        )
        print("Model saved")

        print("\n" + "=" * 60)
        print("Verifying download integrity")
        print("=" * 60)
        verify_download(local_dir)

        return processor, model

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        print("Run script again to resume from breakpoint")
        return None, None

    except Exception as e:
        print(f"\n\nDownload failed: {str(e)}")
        print("\nTroubleshooting suggestions:")
        print("1. Check if network connection is stable")
        print("2. Ensure sufficient disk space (at least 20GB)")
        print("3. If memory is insufficient, close other programs")
        print("4. For specific regions, set mirror: export HF_ENDPOINT=https://hf-mirror.com")
        print("5. Run script again to auto-resume")
        raise


def verify_download(local_dir):
    local_path = Path(local_dir)

    config_files = {
        "config.json": "Model Config",
        "tokenizer_config.json": "Tokenizer Config",
        "preprocessor_config.json": "Preprocessor Config"
    }

    print("\nConfig files:")
    all_configs_ok = True
    for file, desc in config_files.items():
        if (local_path / file).exists():
            size = (local_path / file).stat().st_size / 1024
            print(f"  {desc}: {file} ({size:.1f} KB)")
        else:
            print(f"  {desc}: {file} - Missing")
            all_configs_ok = False

    weight_files = sorted(local_path.glob("*.safetensors")) + \
                   sorted(local_path.glob("*.bin"))

    print(f"\nModel weight files: (Total {len(weight_files)})")
    total_size = 0
    for wf in weight_files:
        size_mb = wf.stat().st_size / (1024**2)
        total_size += size_mb
        print(f"  {wf.name} ({size_mb:.0f} MB)")

    print(f"\nTotal size: {total_size/1024:.2f} GB")

    if all_configs_ok and len(weight_files) >= 4:
        print("\n" + "=" * 60)
        print("Download complete! Model is ready to use")
        print("=" * 60)
        return True
    else:
        print("\nDownload incomplete, please re-run script")
        return False


def load_local_model(local_dir="/path/to/your/own/models/Lingshu-7B"):
    print(f"Loading model from local: {local_dir}")

    if not Path(local_dir).exists():
        print(f"Directory does not exist: {local_dir}")
        print("Please run download script first")
        return None, None

    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            local_dir,
            trust_remote_code=True
        )

        print("Loading model...")
        model = AutoModelForVision2Seq.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        print("Model loaded successfully")
        return processor, model

    except Exception as e:
        print(f"Load failed: {e}")
        return None, None


if __name__ == "__main__":
    import sys

    default_path = "/path/to/your/own/models/Lingshu-7B"

    if len(sys.argv) > 1:
        if sys.argv[1] == "verify":
            verify_download(default_path)
        elif sys.argv[1] == "load":
            load_local_model(default_path)
    else:
        try:
            processor, model = download_lingshu_model_optimized()

            if processor and model:
                print("\nAll done!")


        except KeyboardInterrupt:
            print("\n\nProgram stopped")
        except Exception as e:
            print(f"\nProgram exception: {e}")
            sys.exit(1)