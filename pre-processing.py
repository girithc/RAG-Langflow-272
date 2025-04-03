from langchain_text_splitters import RecursiveCharacterTextSplitter
from langflow.custom import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output
from langflow.schema import Data, DataFrame
from langflow.utils.util import unescape_string


class OptimizedSplitTextComponent(Component):
    display_name: str = "Optimized Split Text"
    description: str = "Efficiently split structured review data into meaningful chunks."
    icon = "scissors-line-dashed"
    name = "OptimizedSplitText"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input Reviews",
            info="Structured product review data.",
            input_types=["Data", "DataFrame"],
            required=True,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info="Maximum number of characters per chunk.",
            value=1000,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of overlapping characters between chunks.",
            value=50,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info="Primary split character (e.g., double newline for structured reviews).",
            value="\n\n",
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="chunks", method="split_text"),
        Output(display_name="DataFrame", name="dataframe", method="as_dataframe"),
    ]

    def _docs_to_data(self, docs) -> list[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def _docs_to_dataframe(self, docs):
        data_dicts = [{"chunk": doc.page_content, **doc.metadata} for doc in docs]
        return DataFrame(data_dicts)

    def split_text_base(self):
        separator = unescape_string(self.separator)
        if isinstance(self.data_inputs, DataFrame):
            if self.data_inputs.empty:
                raise ValueError("DataFrame is empty")
            
            documents = self.data_inputs.to_lc_documents()
        else:
            if not self.data_inputs:
                raise ValueError("No data inputs provided")

            if isinstance(self.data_inputs, Data):
                documents = [self.data_inputs.to_lc_document()]
            else:
                documents = [input_.to_lc_document() for input_ in self.data_inputs if isinstance(input_, Data)]
                if not documents:
                    raise ValueError("No valid Data inputs found")

        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[separator, "\n", " "]  # Prioritizes structured review boundaries
            )
            return splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Error splitting text: {e}") from e

    def split_text(self) -> list[Data]:
        return self._docs_to_data(self.split_text_base())

    def as_dataframe(self) -> DataFrame:
        return self._docs_to_dataframe(self.split_text_base())
