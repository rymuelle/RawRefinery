import os
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def main():
    directory = input("Enter directory containing files to sign: ").strip()

    directory = Path(directory).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        print("ERROR: Not a valid directory.")
        return

    private_key_path = "model_signing_private_key.pem"
    private_key = serialization.load_pem_private_key(
        open(private_key_path, "rb").read(),
        password=None,
    )

    # Loop through all files in directory
    for file_path in directory.iterdir():
        if file_path.is_file() and not file_path.name.endswith(".sig"):
            print(f"Signing: {file_path.name}")

            data = file_path.read_bytes()
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            sig_path = file_path.with_suffix(file_path.suffix + ".sig")
            sig_path.write_bytes(signature)

            print(f" → Created signature: {sig_path.name}")

    print("\nDone!")

if __name__ == "__main__":
    main()
