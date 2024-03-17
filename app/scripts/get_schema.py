import os
import sqlite3

sqlite_uri = "./data/Chinook.db"

# get database schema to a file
if not os.path.exists(sqlite_uri):
    print("not file")
else:
    print("ok")
with sqlite3.connect(sqlite_uri) as conn:
    with open("./data/schema.sql", "w") as f:
        for line in conn.iterdump():
            if line.startswith("INSERT"):
                continue
            f.write(line + "\n")
            print(line)
            print()
