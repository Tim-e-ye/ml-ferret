CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Go Board Constants (野狐原生 A-S 规范，共 19 列)
GO_BOARD_SIZE = 19
GO_COLS = "ABCDEFGHIJKLMNOPQRS"
GO_POSITION_TOKEN_TEMPLATE = "<go_{}>"  # e.g., <go_A1>, ..., <go_S19>
GO_POSITION_TOKENS = [
    GO_POSITION_TOKEN_TEMPLATE.format(f"{col}{row}")
    for row in range(1, GO_BOARD_SIZE + 1)
    for col in GO_COLS
]  # 361 个专用 Token: ["<go_A1>", "<go_B1>", ..., "<go_S19>"]

