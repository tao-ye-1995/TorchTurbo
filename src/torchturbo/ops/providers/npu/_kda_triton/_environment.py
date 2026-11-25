"""Check the CANN/compiler ABI used by the frozen K3 kernels."""
import json
import os
from pathlib import Path
import subprocess


def configure():
    assets = Path(__file__).resolve().parent / '_ir'
    manifest = json.loads((assets / 'manifest.json').read_text())
    home = Path(os.environ.get('ASCEND_HOME_PATH', '')).resolve()
    info = home / 'share/info/runtime/version.info'
    if not info.is_file() or f"Version={manifest['cann_version']}" not in info.read_text().splitlines():
        raise RuntimeError('kda-triton requires CANN 9.1.0; source its set_env.sh first')
    compiler = home / 'tools/bishengir/bin/bishengir-compile'
    os.environ['PATH'] = str(compiler.parent) + os.pathsep + os.environ['PATH']
    from triton.backends.ascend.utils import _get_npucompiler_path
    import triton
    actual = Path(_get_npucompiler_path()[0]).resolve()
    if actual != compiler.resolve():
        raise RuntimeError(f'Triton resolves {actual}, expected {compiler}')
    version = subprocess.check_output([str(actual), '--version'], text=True)
    if version != manifest['compiler_version'] or triton.__version__ != manifest['triton_version']:
        raise RuntimeError('Frozen K3 requires the recorded BiSheng 1.2.0 and Triton 3.5.1 ABI')
    # Cache isolation applies to JIT K1/K2 and the Triton launch adapters.
    if 'TRITON_CACHE_DIR' not in os.environ:
        os.environ['TRITON_CACHE_DIR'] = str(Path.home() / '.cache/kda_triton/cann9.1_bisheng1.2_triton3.5.1')
    return dict(cann=str(home), latest=str(Path('/usr/local/Ascend/ascend-toolkit/latest').resolve()),
                compiler=str(actual), compiler_version=version, target=manifest['target'],
                cache=os.environ['TRITON_CACHE_DIR'])
