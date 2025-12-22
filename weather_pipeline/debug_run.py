import runpy, glob, traceback
print('files:', glob.glob('tests/test_*.py'))
try:
    runpy.run_path('tests/test_validators.py', run_name='__main__')
    print('ran')
except Exception:
    traceback.print_exc()
