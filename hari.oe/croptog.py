from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
sensitive_data = b"User Sensitive Data"
encrypted_data = cipher_suite.encrypt(sensitive_data)
print(f"Encrypted: {encrypted_data}")
decrypted_data = cipher_suite.decrypt(encrypted_data)
print(f"Decrypted: {decrypted_data.decode('utf-8')}")