from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# Special Tokens
# ============================================================

SPECIAL_TOKENS = [
    "PAD",
    "BOS",
    "EOS",
]


# ============================================================
# Code Vocabulary
# ============================================================

CODE_VOCAB = [
    "PRINT",
    "LPAREN",
    "RPAREN",

    "NAME_X",
    "NAME_Y",
    "NAME_A",
    "NAME_B",

    "STR_HELLO",

    "NUM_0",
    "NUM_1",

    "ASSIGN",
    "PLUS",
    "MINUS",
    "DIVIDE",

    "NEWLINE",
]


# ============================================================
# Vocabulary Class
# ============================================================

@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def build(
        cls,
        tokens: list[str],
    ) -> "Vocabulary":

        unique_tokens = []

        for token in tokens:

            if token not in unique_tokens:

                unique_tokens.append(
                    token
                )


        token_to_id = {
            token: index
            for index, token
            in enumerate(unique_tokens)
        }


        id_to_token = {
            index: token
            for token, index
            in token_to_id.items()
        }


        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )


    # ========================================================
    # Encode tokens -> IDs
    # ========================================================

    def encode(
        self,
        tokens: list[str],
    ) -> list[int]:

        ids = []

        for token in tokens:

            if token not in self.token_to_id:

                raise KeyError(
                    f"Unknown token: {token}"
                )

            ids.append(
                self.token_to_id[token]
            )


        return ids


    # ========================================================
    # Decode IDs -> tokens
    # ========================================================

    def decode(
        self,
        ids: list[int],
    ) -> list[str]:

        tokens = []

        for token_id in ids:

            if token_id not in self.id_to_token:

                raise KeyError(
                    f"Unknown token ID: "
                    f"{token_id}"
                )

            tokens.append(
                self.id_to_token[
                    token_id
                ]
            )


        return tokens


    # ========================================================
    # Vocabulary Size
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.token_to_id
        )


# ============================================================
# Prompt Vocabulary
# ============================================================

def build_prompt_vocabulary(
    prompt_tokens: list[str],
) -> Vocabulary:

    tokens = (
        SPECIAL_TOKENS
        +
        sorted(
            set(prompt_tokens)
        )
    )


    return Vocabulary.build(
        tokens
    )


# ============================================================
# Code Vocabulary
# ============================================================

def build_code_vocabulary() -> Vocabulary:

    tokens = (
        SPECIAL_TOKENS
        +
        CODE_VOCAB
    )


    return Vocabulary.build(
        tokens
    )


# ============================================================
# Python Code -> Semantic Tokens
# ============================================================

def code_to_tokens(
    code: str,
) -> list[str]:

    mappings = {

        # ----------------------------------------------------
        # print("hello")
        # ----------------------------------------------------

        'print("hello")': [
            "PRINT",
            "LPAREN",
            "STR_HELLO",
            "RPAREN",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # print(x)
        # ----------------------------------------------------

        "print(x)": [
            "PRINT",
            "LPAREN",
            "NAME_X",
            "RPAREN",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # x = 0
        # ----------------------------------------------------

        "x = 0": [
            "NAME_X",
            "ASSIGN",
            "NUM_0",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # x = 1
        # ----------------------------------------------------

        "x = 1": [
            "NAME_X",
            "ASSIGN",
            "NUM_1",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # x = a
        # ----------------------------------------------------

        "x = a": [
            "NAME_X",
            "ASSIGN",
            "NAME_A",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # a = x
        # ----------------------------------------------------

        "a = x": [
            "NAME_A",
            "ASSIGN",
            "NAME_X",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # y = a + b
        # ----------------------------------------------------

        "y = a + b": [
            "NAME_Y",
            "ASSIGN",
            "NAME_A",
            "PLUS",
            "NAME_B",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # y = a - b
        # ----------------------------------------------------

        "y = a - b": [
            "NAME_Y",
            "ASSIGN",
            "NAME_A",
            "MINUS",
            "NAME_B",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # y = b - a
        # ----------------------------------------------------

        "y = b - a": [
            "NAME_Y",
            "ASSIGN",
            "NAME_B",
            "MINUS",
            "NAME_A",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # y = a / b
        # ----------------------------------------------------

        "y = a / b": [
            "NAME_Y",
            "ASSIGN",
            "NAME_A",
            "DIVIDE",
            "NAME_B",
            "NEWLINE",
            "EOS",
        ],


        # ----------------------------------------------------
        # y = b / a
        # ----------------------------------------------------

        "y = b / a": [
            "NAME_Y",
            "ASSIGN",
            "NAME_B",
            "DIVIDE",
            "NAME_A",
            "NEWLINE",
            "EOS",
        ],
    }


    if code not in mappings:

        raise ValueError(
            f"Unknown code template: "
            f"{code}"
        )


    return mappings[
        code
    ].copy()


# ============================================================
# Semantic Tokens -> Python Code
# ============================================================

def tokens_to_code(
    tokens: list[str],
) -> str:

    # Remove tokens that are not part of the final
    # source-code representation.

    filtered = [
        token
        for token in tokens
        if token not in {
            "PAD",
            "BOS",
            "EOS",
            "NEWLINE",
        }
    ]


    key = tuple(
        filtered
    )


    mappings = {

        # ----------------------------------------------------
        # print("hello")
        # ----------------------------------------------------

        (
            "PRINT",
            "LPAREN",
            "STR_HELLO",
            "RPAREN",
        ): 'print("hello")',


        # ----------------------------------------------------
        # print(x)
        # ----------------------------------------------------

        (
            "PRINT",
            "LPAREN",
            "NAME_X",
            "RPAREN",
        ): "print(x)",


        # ----------------------------------------------------
        # x = 0
        # ----------------------------------------------------

        (
            "NAME_X",
            "ASSIGN",
            "NUM_0",
        ): "x = 0",


        # ----------------------------------------------------
        # x = 1
        # ----------------------------------------------------

        (
            "NAME_X",
            "ASSIGN",
            "NUM_1",
        ): "x = 1",


        # ----------------------------------------------------
        # x = a
        # ----------------------------------------------------

        (
            "NAME_X",
            "ASSIGN",
            "NAME_A",
        ): "x = a",


        # ----------------------------------------------------
        # a = x
        # ----------------------------------------------------

        (
            "NAME_A",
            "ASSIGN",
            "NAME_X",
        ): "a = x",


        # ----------------------------------------------------
        # y = a + b
        # ----------------------------------------------------

        (
            "NAME_Y",
            "ASSIGN",
            "NAME_A",
            "PLUS",
            "NAME_B",
        ): "y = a + b",


        # ----------------------------------------------------
        # y = a - b
        # ----------------------------------------------------

        (
            "NAME_Y",
            "ASSIGN",
            "NAME_A",
            "MINUS",
            "NAME_B",
        ): "y = a - b",


        # ----------------------------------------------------
        # y = b - a
        # ----------------------------------------------------

        (
            "NAME_Y",
            "ASSIGN",
            "NAME_B",
            "MINUS",
            "NAME_A",
        ): "y = b - a",


        # ----------------------------------------------------
        # y = a / b
        # ----------------------------------------------------

        (
            "NAME_Y",
            "ASSIGN",
            "NAME_A",
            "DIVIDE",
            "NAME_B",
        ): "y = a / b",


        # ----------------------------------------------------
        # y = b / a
        # ----------------------------------------------------

        (
            "NAME_Y",
            "ASSIGN",
            "NAME_B",
            "DIVIDE",
            "NAME_A",
        ): "y = b / a",
    }


    if key not in mappings:

        raise ValueError(
            "Cannot decode token sequence: "
            f"{tokens}"
        )


    return mappings[
        key
    ]


# ============================================================
# Teacher-Forcing Helper
# ============================================================

def build_teacher_forcing_pairs(
    code: str,
) -> list[
    tuple[
        list[str],
        str,
    ]
]:

    """
    Convert target code into teacher-forcing examples.

    Example:

        y = a - b

    becomes:

        [BOS] -> NAME_Y

        [BOS, NAME_Y]
            -> ASSIGN

        [BOS, NAME_Y, ASSIGN]
            -> NAME_A

        ...

        [..., NEWLINE]
            -> EOS
    """

    target_tokens = code_to_tokens(
        code
    )


    context = [
        "BOS"
    ]


    pairs = []


    for target_token in target_tokens:

        pairs.append(
            (
                context.copy(),
                target_token,
            )
        )


        # EOS ends generation.
        if target_token == "EOS":

            break


        context.append(
            target_token
        )


    return pairs