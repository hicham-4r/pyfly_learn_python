from pyfly.dataset import load_dataset

from pyfly.tokenizer import (
    build_code_vocabulary,
    code_to_tokens,
    tokens_to_code,
    build_teacher_forcing_pairs,
)


DATASETS = [
    "data/raw/python_mre.jsonl",
    "data/raw/sequence_challenge.jsonl",
]


code_vocab = (
    build_code_vocabulary()
)


print()
print("========================================")
print("       PyFly GENERATOR TOKEN TEST")
print("========================================")
print()

print(
    f"Code vocabulary size: "
    f"{len(code_vocab)}"
)

print()


total = 0


for dataset_path in DATASETS:

    examples = load_dataset(
        dataset_path
    )


    print()
    print(
        f"Dataset: "
        f"{dataset_path}"
    )

    print()


    for example in examples:

        tokens = code_to_tokens(
            example.code
        )


        ids = code_vocab.encode(
            tokens
        )


        decoded_tokens = (
            code_vocab.decode(
                ids
            )
        )


        decoded_code = (
            tokens_to_code(
                decoded_tokens
            )
        )


        assert (
            decoded_code
            == example.code
        )


        assert (
            tokens[-1]
            == "EOS"
        )


        assert (
            decoded_tokens
            == tokens
        )


        teacher_pairs = (
            build_teacher_forcing_pairs(
                example.code
            )
        )


        assert (
            len(teacher_pairs)
            == len(tokens)
        )


        assert (
            teacher_pairs[0][0]
            == ["BOS"]
        )


        assert (
            teacher_pairs[-1][1]
            == "EOS"
        )


        total += 1


        print(
            f"✓ "
            f"{example.code:15} "
            f"-> "
            f"{tokens}"
        )


print()
print("========================================")

print(
    f"Tested examples: "
    f"{total}"
)

print(
    f"Vocabulary size: "
    f"{len(code_vocab)}"
)

print("========================================")

print()
print(
    "✅ Generator token vocabulary PASSED"
)