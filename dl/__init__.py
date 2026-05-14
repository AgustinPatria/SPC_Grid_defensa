"""Paquete `dl` — capa de Deep Learning del proyecto.

La configuracion vive en `config.py` (raiz del proyecto). Para no romper
checkpoints previos que fueron picklados con el path antiguo
`dl.config.DLConfig`, aqui redirigimos `dl.config` al modulo `config` raiz.
"""

import sys as _sys

import config as _config

_sys.modules.setdefault("dl.config", _config)
