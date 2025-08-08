from cx_Freeze import setup, Executable

setup(name="NLP model voice conversation", executables=[Executable("NLP model voice conversation script.py")], options={"build_exe": {"excludes": ["tkinter"]}})
