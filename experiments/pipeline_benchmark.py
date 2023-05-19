import json
import os
import pathlib
import shutil
import subprocess
import time


def run_benchmarks(folder_path, save_folder):
    # Liste zum Speichern der gefundenen Python-Skripte
    python_files = []

    # Durchlaufen des Ordners und seiner Unterordner
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # Überprüfen, ob die Datei die Endung ".py" hat
            if filename.startswith("EXP") and filename.endswith(".py"):
                # Absoluten Pfad zur Datei erstellen
                file_path = os.path.join(dirpath, filename)
                python_files.append(file_path)

    # Ausführen der gefundenen Python-Skripte
    for python_file in python_files:
        try:
            # Ausführen des Python-Skripts
            python_exe = "C:/Users/Leonie Freisinger/Leonie/01_Dokumente/02_Studium/Master/01_Masterstudium/NeuralProphet/tot4/Scripts/python.exe"
            subprocess.run([python_exe, python_file], check=True)
        except subprocess.CalledProcessError as e:
            # Fehlerbehandlung, falls das Skript fehlschlägt
            print(f"Skript {python_file} ist fehlgeschlagen mit Rückgabecode {e.returncode}")

            # Save the aborted file to the specified save folder
            filename = os.path.basename(python_file)
            save_path = os.path.join(save_folder, filename)
            shutil.copy(python_file, save_path)


def get_experiment_scripts(folder_path):
    # Liste zum Speichern der gefundenen Python-Skripte
    python_files = []

    # Durchlaufen des Ordners und seiner Unterordner
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # Überprüfen, ob die Datei die Endung ".py" hat
            if filename.startswith("EXP") and filename.endswith(".py"):
                # Absoluten Pfad zur Datei erstellen
                file_path = os.path.join(dirpath, filename)
                python_files.append(file_path)
    return python_files


# Pfad zum Hauptordner
dir = pathlib.Path(__file__).parent.absolute()
scripts_dir = os.path.join(dir, "scripts")
print(dir)
# Ausführen der Python-Skripte in den Unterordnern
start_time = time.time()
run_benchmarks(scripts_dir, dir)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
# save elapsed time
time_file_name = os.path.join(dir, f"total_time.json")
with open(time_file_name, "w") as file:
    json.dump(elapsed_time, file)

# print(get_experiment_scripts(scripts_dir))
