import functools
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import time


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


def run_python_file(python_exe, python_file, save_folder):
    try:
        # Execute the Python script
        # python_exe = "C:/Users/Leonie Freisinger/Leonie/01_Dokumente/02_Studium/Master/01_Masterstudium/NeuralProphet/tot4/Scripts/python.exe"
        subprocess.run([python_exe, python_file], check=True)
    except subprocess.CalledProcessError as e:
        # Handling for failed scripts
        print(f"Skript {python_file} ist fehlgeschlagen mit Rückgabecode {e.returncode}")

        # Save the aborted file to the specified save folder
        filename = os.path.basename(python_file)
        save_path = os.path.join(save_folder, filename)
        shutil.copy(python_file, save_path)


def run_script(args):
    python_file, save_folder = args
    print(f"Running {python_file} in process id {os.getpid()}")
    try:
        # Ausführen des Python-Skripts
        # python_exe = "C:/Users/Leonie Freisinger/Leonie/01_Dokumente/02_Studium/Master/01_Masterstudium/NeuralProphet/tot4/Scripts/python.exe"
        python_exe = "python3"  # for runnin gon ubuntu
        subprocess.run([python_exe, python_file], check=True)
    except subprocess.CalledProcessError as e:
        # Fehlerbehandlung, falls das Skript fehlschlägt
        print(f"Skript {python_file} ist fehlgeschlagen mit Rückgabecode {e.returncode}")

        # Save the aborted file to the specified save folder
        filename = os.path.basename(python_file)
        save_path = os.path.join(save_folder, filename)
        shutil.copy(python_file, save_path)


def run_benchmarks(folder_path, save_folder):
    # Liste zum Speichern der gefundenen Python-Skripte
    python_files = get_experiment_scripts(folder_path)

    # Create a pool of processes
    with multiprocessing.Pool(processes=4) as pool:
        # Use starmap function with list of tuples, where each tuple contains parameters for run_script
        pool.map(run_script, [(python_file, save_folder) for python_file in python_files])


# Ausführen der Python-Skripte in den Unterordnern
start_time = time.time()
dir_path = pathlib.Path(__file__).parent.absolute()
scripts_dir = os.path.join(dir_path, "scripts")
print(dir_path)
if __name__ == "__main__":
    # Specify the Python interpreter path and folders for benchmarks and saving aborted files
    run_benchmarks(scripts_dir, dir_path)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
# save elapsed time
time_file_name = os.path.join(dir_path, f"total_time.json")
with open(time_file_name, "w") as file:
    json.dump(elapsed_time, file)

# print(get_experiment_scripts(scripts_dir))
