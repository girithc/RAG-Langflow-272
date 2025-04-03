from langflow.custom import Component
from langflow.helpers.data import data_to_text, data_to_text_list
from langflow.io import DataInput, MultilineInput, Output, StrInput
from langflow.schema import Data
from langflow.schema.message import Message


class OptimizedParseDataComponent(Component):
    display_name = "Structured Data to Message"
    description = "Convert structured review data into formatted messages."
    icon = "message-square"
    name = "OptimizedParseData"

    inputs = [
        DataInput(
            name="data",
            display_name="Review Data",
            info="Structured product review data.",
            is_list=True,
            required=True,
        ),
        MultilineInput(
            name="template",
            display_name="Template",
            info=(
                "Custom formatting for review data. Use placeholders like {text}, {Product ID}, {Rating}."
            ),
            value="Review for {Product ID}: Rated {Rating}/5. \nUser said: {text}",
            required=True,
        ),
        StrInput(name="sep", display_name="Separator", advanced=True, value="\n\n"),
    ]

    outputs = [
        Output(
            display_name="Formatted Message",
            name="text",
            info="Structured data as a formatted message.",
            method="parse_data",
        ),
        Output(
            display_name="Data List",
            name="data_list",
            info="List of formatted review data entries.",
            method="parse_data_as_list",
        ),
    ]

    def _clean_args(self) -> tuple[list[Data], str, str]:
        data = self.data if isinstance(self.data, list) else [self.data]
        template = self.template
        sep = self.sep
        return data, template, sep

    def parse_data(self) -> Message:
        data, template, sep = self._clean_args()
        formatted_text = data_to_text(template, data, sep)
        self.status = formatted_text
        return Message(text=formatted_text)

    def parse_data_as_list(self) -> list[Data]:
        data, template, _ = self._clean_args()
        text_list, data_list = data_to_text_list(template, data)
        for item, text in zip(data_list, text_list, strict=True):
            item.set_text(text)
        self.status = data_list
        return data_list
