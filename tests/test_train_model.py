import unittest
from unittest import mock
from unittest.mock import patch, MagicMock
from transformers import CLIPTokenizer
from datasets import load_dataset
import torch
from gen_ai_photoshoots.train_model import DreamBoothDataset, collate_fn


class TestDreamBoothDataset(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset and tokenizer
        self.dataset = load_dataset("imagefolder", data_dir="tests/testing_photos", split="train")
        self.instance_prompt = "a photo of a cat"
        self.tokenizer = MagicMock(spec=CLIPTokenizer)
        self.tokenizer.model_max_length = 77
        self.tokenizer.return_value = {"input_ids": [1, 2, 3]}

        self.dataset = DreamBoothDataset(self.dataset, self.instance_prompt, self.tokenizer)

    def test_len(self):
        self.assertEqual(len(self.dataset), 6)

    @patch('gen_ai_photoshoots.train_model.transforms.ToTensor')
    def test_getitem(self, mock_transforms):
        mock_transforms.return_value = MagicMock()
        example = self.dataset[0]
        self.assertIn("instance_images", example)
        self.assertIn("instance_prompt_ids", example)
        self.assertEqual(example["instance_prompt_ids"], [1, 2, 3])


class TestCollateFn(unittest.TestCase):

    @mock.patch('gen_ai_photoshoots.train_model.CLIPTokenizer.from_pretrained')
    def test_collate_fn(self, mock_tokenizer_from_pretrained):
        # Mocking CLIPTokenizer.from_pretrained
        mock_tokenizer = MagicMock(spec=CLIPTokenizer)
        mock_tokenizer.pad.return_value = {"input_ids": torch.tensor([0]), "attention_mask": torch.tensor([1])}
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Creating example data with tensors instead of MagicMock objects
        examples = [
            {"instance_prompt_ids": torch.tensor([1, 2, 3]), "instance_images": torch.randn(3, 224, 224)}
        ]

        # Calling the collate function
        batch = collate_fn(examples)

        # Assertions
        # Ensure collate_fn returns a dictionary with the expected keys
        self.assertTrue(isinstance(batch, dict))
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("pixel_values", batch)

        # Make sure the tokenizer is called with the correct arguments
        mock_tokenizer.pad.assert_called_once_with(
            {"input_ids": [example["instance_prompt_ids"] for example in examples]},
            padding=True, return_attention_mask=True, return_tensors="pt"
        )

        # Make sure the image inputs are stacked correctly
        self.assertTrue(
            isinstance(batch["pixel_values"], torch.Tensor)
            and batch["pixel_values"].shape[0] == len(examples)
        )

        # Make sure all other keys in the batch are of type torch.Tensor
        for key in batch.keys():
            if key not in ["pixel_values"]:
                self.assertTrue(isinstance(batch[key], torch.Tensor))
