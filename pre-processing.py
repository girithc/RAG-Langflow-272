from langchain_text_splitters import CharacterTextSplitter
from langflow.custom import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output
from langflow.schema import Data, DataFrame
from langflow.utils.util import unescape_string
from typing import List, Union

class SplitTextComponent(Component):
    display_name: str = "Split Text (Labeled Reviews)"
    description: str = "Split labeled review text into chunks (handles __label__X prefix)."
    icon = "scissors-line-dashed"
    name = "SplitLabeledText"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input Documents",
            info="The raw text data with __label__ prefixes to split.",
            input_types=["Data", "DataFrame", "str"],
            required=True,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of characters to overlap between chunks.",
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info="The maximum number of characters in each chunk.",
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info="The character to split on. Defaults to newline.",
            value="\n",
        ),
        MessageTextInput(
            name="label_prefix",
            display_name="Label Prefix",
            info="Prefix for labels (e.g., '__label__')",
            value="__label__",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="chunks", method="split_text"),
        Output(display_name="DataFrame", name="dataframe", method="as_dataframe"),
    ]

    def _preprocess_text(self, text: str) -> List[dict]:
        """Extract labels and text from raw input."""
        lines = text.split(self.separator)
        processed = []
        for line in lines:
            if not line.strip():
                continue
            # Split at first occurrence of label prefix
            if self.label_prefix in line:
                prefix, content = line.split(self.label_prefix, 1)
                label, text = content.split(" ", 1)
                processed.append({
                    "text": text.strip(),
                    "label": label.strip(),
                    "raw": line.strip()
                })
        return processed

    def _docs_to_data(self, docs) -> List[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def _docs_to_dataframe(self, docs):
        data_dicts = []
        for doc in docs:
            data = {"text": doc.page_content}
            if hasattr(doc, "metadata"):
                data.update(doc.metadata)
            data_dicts.append(data)
        return DataFrame(data_dicts)

    def split_text_base(self):
        separator = unescape_string(self.separator)
        
        # Handle raw string input
        if isinstance(self.data_inputs, str):
            processed = self._preprocess_text(self.data_inputs)
            documents = [Data(text=item["text"], data={"label": item["label"]}) for item in processed]
        
        # Handle DataFrame input
        elif isinstance(self.data_inputs, DataFrame):
            if len(self.data_inputs) == 0:
                raise ValueError("DataFrame is empty")
            documents = self.data_inputs.to_lc_documents()
        
        # Handle Data input
        else:
            if not self.data_inputs:
                raise ValueError("No data inputs provided")
            
            if isinstance(self.data_inputs, Data):
                documents = [self.data_inputs.to_lc_document()]
            else:
                documents = []
                for item in self.data_inputs:
                    if isinstance(item, Data):
                        documents.append(item.to_lc_document())
                    elif isinstance(item, str):
                        processed = self._preprocess_text(item)
                        documents.extend([Data(text=p["text"], data={"label": p["label"]}) for p in processed])
                
                if not documents:
                    raise ValueError("No valid data found in inputs")

        # Split documents
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=separator
        )
        return splitter.split_documents(documents)

    def split_text(self) -> List[Data]:
        return self._docs_to_data(self.split_text_base())

    def as_dataframe(self) -> DataFrame:
        return self._docs_to_dataframe(self.split_text_base())
