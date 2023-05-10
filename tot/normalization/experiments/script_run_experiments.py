
import pathlib
import os
import subprocess
import time

def run_python_scripts_in_subfolders(folder_path):
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

# Pfad zum Hauptordner
folder_path = pathlib.Path(__file__).parent.parent.absolute()
print(folder_path)
# Ausführen der Python-Skripte in den Unterordnern
start_time = time.time()
run_python_scripts_in_subfolders(folder_path)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)