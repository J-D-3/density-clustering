# Third-Party Licenses

OPTICS-Clustering itself is licensed under the MIT License (see `LICENSE`). It
bundles the third-party components below. Their licenses are reproduced here;
the copyright notices are retained in the vendored source files as well.

| Component | Where | License | Scope |
|-----------|-------|---------|-------|
| nanoflann | `include/optics/nanoflann.hpp` | BSD 2-Clause | **runtime** (default neighbor-search backend) |
| hnswlib   | `include/optics/hnswlib/`      | Apache-2.0   | **runtime, opt-in** (HNSW backend, only with `OPTICS_ENABLE_HNSW`) |
| doctest   | `test/third_party/doctest.h`   | MIT          | test-only |
| nanobench | `test/third_party/nanobench.h` | MIT          | test-only (perf harness) |

Only **nanoflann** is part of the default library you compile against. **hnswlib** is
vendored but compiled only when the optional `OPTICS_ENABLE_HNSW` backend is enabled (off
by default). doctest and nanobench are used solely to build the tests/benchmarks and are
not required to use OPTICS-Clustering.

---

## nanoflann — BSD 2-Clause License

Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
Copyright 2011-2024  Jose Luis Blanco (joseluisblancoc@gmail.com). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## hnswlib — Apache License 2.0

Copyright Yury Malkov and the hnswlib contributors (https://github.com/nmslib/hnswlib).

Licensed under the Apache License, Version 2.0. The full license text is vendored
alongside the headers at `include/optics/hnswlib/LICENSE` and is also available at
<http://www.apache.org/licenses/LICENSE-2.0>. hnswlib is compiled into the library only
when the optional `OPTICS_ENABLE_HNSW` backend is enabled (off by default); the default
build does not include it.

---

## doctest — MIT License

Copyright (c) 2016-2023 Viktor Kirilov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## nanobench — MIT License

Copyright (c) 2019-2023 Martin Leitner-Ankerl <martin.ankerl@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
