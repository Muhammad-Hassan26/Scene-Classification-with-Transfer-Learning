
import io

import torch
import torch.nn.functional as F

from fastapi.testclient import TestClient
from PIL import Image

import app as api

SAMPLE = "sample.jpg"


def make_sample_if_missing():
    try:
        with open(SAMPLE, "rb") as f:
            f.read(1)
    except FileNotFoundError:
        Image.new(
            "RGB", (224, 224), "white"
        ).save(SAMPLE, "JPEG")


def post_image(client):
    with open(SAMPLE, "rb") as f:
        return client.post(
            "/predict",
            files={
                "file": (
                    "sample.jpeg",
                    f,
                    "image/jpeg"
                )
            }
        )


# TEST 1: Health endpoint

def test_health():

    with TestClient(api.app) as c:

        r = c.get("/health")

        assert r.status_code == 200
        assert r.json()["model_loaded"] is True
        assert len(r.json()["classes"]) == 4


# TEST 2: Prediction and probabilities

def test_predict_schema_and_probabilities():

    make_sample_if_missing()

    with TestClient(api.app) as c:

        r = post_image(c)

        assert r.status_code == 200

        probs = r.json()["probabilities"]

        assert abs(sum(probs.values()) - 1) < 1e-3


# TEST 3: Direct model parity

def test_parity_with_direct_model():

    make_sample_if_missing()

    with TestClient(api.app) as c:

        served = post_image(c).json()["probabilities"]

        ck = api.state["ck"]
        tf = api.state["tf"]

        img = Image.open(SAMPLE).convert(
            "L" if ck["grayscale"] else "RGB"
        )

        with torch.inference_mode():

            p = F.softmax(
                api.state["model"](
                    tf(img).unsqueeze(0)
                ),
                1
            )[0]

        direct = dict(zip(ck["classes"], p.tolist()))

        for k in direct:

            assert abs(
                served[k] - direct[k]
            ) < 1e-3


# TEST 4: Reject wrong file type

def test_rejects_wrong_type():

    with TestClient(api.app) as c:

        r = c.post(
            "/predict",
            files={
                "file": (
                    "a.txt",
                    b"hello",
                    "text/plain"
                )
            }
        )

        assert r.status_code == 415


# TEST 5: Reject oversized image

def test_rejects_too_large():

    big = io.BytesIO(
        b"0" * (api.MAX_BYTES + 1)
    )

    with TestClient(api.app) as c:

        r = c.post(
            "/predict",
            files={
                "file": (
                    "big.jpg",
                    big,
                    "image/jpeg"
                )
            }
        )

        assert r.status_code == 413


# TEST 6: Reject corrupt image

def test_rejects_corrupt_image():

    with TestClient(api.app) as c:

        r = c.post(
            "/predict",
            files={
                "file": (
                    "bad.jpg",
                    b"not an image",
                    "image/jpeg"
                )
            }
        )

        assert r.status_code == 400


# TEST 7: Missing file

def test_missing_file_is_422():

    with TestClient(api.app) as c:

        assert c.post("/predict").status_code == 422


# TEST 8: PII masking

def test_analyze_masks_pii():

    make_sample_if_missing()

    note = (
        "Brown lesions on leaves. "
        "Contact 0300-1234567, "
        "CNIC 35202-1234567-1"
    )

    with TestClient(api.app) as c:

        with open(SAMPLE, "rb") as f:

            r = c.post(
                "/analyze",
                files={
                    "file": (
                        "sample.jpeg",
                        f,
                        "image/jpeg"
                    )
                },
                data={"note": note}
            )

        body = r.json()

        assert r.status_code == 200

        assert "0300-1234567" not in str(body)

        assert "35202-1234567-1" not in str(body)

        assert body["urgent_terms"]

        assert body["decision"] == "REFER"
