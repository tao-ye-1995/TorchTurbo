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

"""K1 launch interface for tensor pointers, runtime integers and scalar options."""
import triton


class _PrepareLauncher:
    """Tensor arguments retain dtype and 16-byte alignment. Runtime dimensions are integer
    scalars; other options retain their scalar types and values.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.runners = {}

    def __getitem__(self, grid):
        def launch(**kw):
            kernel = self.kernel
            if kernel.pre_run_hooks:
                return kernel[grid](**kw)
            device = triton.runtime.driver.active.get_current_device()
            pointers = tuple((kw[n].dtype, kw[n].data_ptr() % 16 == 0)
                             for n in kernel.arg_names[:9])
            dynamic = tuple('i32' if -(2**31) <= kw[n] < 2**31 else 'u64' if 2**63 <= kw[n] < 2**64 else 'i64'
                            for n in ('T_total','task_num','num_core','HV'))
            scalars = tuple((n,type(v),v) for n,v in kw.items() if n not in kernel.arg_names[:9]
                            and n not in ('T_total','task_num','num_core','HV'))
            key = (device,grid,pointers,dynamic,scalars)
            runner=self.runners.get(key)
            if runner is None:
                compiled=kernel[grid](**kw)
                self.runners[key]=(compiled,compiled[(grid[0],1,1)])
                return compiled
            for (name,_),(value,scope) in kernel.used_global_vals.items():
                if name not in scope or scope[name] != value:
                    raise RuntimeError(f'Global variable {name} has changed since kernel compilation')
            runner[1](*(kw[n] for n in kernel.arg_names))
            return runner[0]
        return launch
