print("Cloud agent running!")

import os
os.makedirs("outputs", exist_ok=True)

with open("outputs/test.txt", "w") as f:
    f.write("success")
