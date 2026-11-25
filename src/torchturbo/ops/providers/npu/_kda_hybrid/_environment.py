# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process environment configuration for the bundled AscendC interface."""
import os
from pathlib import Path
import platform


def configure():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Bundled AscendC K3 requires Linux AArch64')
    home = Path(os.environ.get('ASCEND_HOME_PATH', '')).resolve()
    info = home / 'share/info/runtime/version.info'
    if not info.exists() or 'Version=9.1.0' not in info.read_text().splitlines():
        raise RuntimeError('kda-hybrid requires CANN 9.1.0; source its set_env.sh first')
    compiler = home / 'tools/bishengir/bin/bishengir-compile'
    os.environ['PATH'] = str(compiler.parent) + os.pathsep + os.environ['PATH']
    vendor = Path(__file__).resolve().parent / '_native/opp/vendors/kda_hybrid'
    if not (vendor/'op_api/lib/libcust_opapi.so').is_file():
        raise RuntimeError('Missing bundled AscendC K3 opapi library; reinstall kda-hybrid')
    parts = [str(vendor), *[p for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(os.pathsep)
                          if p and p != str(vendor)]]
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = os.pathsep.join(parts)
    os.environ['KDA_HYBRID_OP_API_LIBS'] = str(vendor/'op_api/lib/libcust_opapi.so')
    from triton.backends.ascend.utils import _get_npucompiler_path
    actual = Path(_get_npucompiler_path()[0]).resolve()
    if actual != compiler.resolve():
        raise RuntimeError(f'FlagTree resolves {actual}, expected CANN compiler {compiler}')
    return dict(cann=str(home), compiler=str(actual), vendor=str(vendor))
