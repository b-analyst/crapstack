import logging
from abc import ABC
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from crapstack.document_stores.base import BaseDocumentStore
from crapstack.haystack.nodes.answer_generator.base import BaseGenerator
from crapstack.haystack.nodes.other.docs2answers import Docs2Answers
from crapstack.haystack.nodes.other.document_merger import DocumentMerger
from crapstack.haystack.nodes.question_generator.question_generator import QuestionGenerator
from crapstack.haystack.nodes.reader.base import BaseReader
from crapstack.haystack.nodes.retriever.base import BaseRetriever
from crapstack.haystack.nodes.preprocessor.preprocessor import PreProcessor
from crapstack.haystack.nodes.file_converter.txt import TextConverter
from crapstack.haystack.pipelines.base import Pipeline
from crapstack.haystack.schema import Document, EvaluationResult, MultiLabel, FilterType


logger = logging.getLogger(__name__)


class BaseStandardPipeline(ABC):
    """
    Base class for pre-made standard Haystack pipelines.
    This class does not inherit from Pipeline.
    """

    pipeline: Pipeline
    metrics_filter: Optional[Dict[str, List[str]]] = None

    def add_node(self, component, name: str, inputs: List[str]):
        """
        Add a new node to the pipeline.

        :param component: The object to be called when the data is passed to the node. It can be a Haystack component
                          (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                          method to process incoming data from predecessor node.
        :param name: The name for the node. It must not contain any dots.
        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
                       of node is sufficient. For instance, a 'BM25Retriever' node would always output a single
                       edge with a list of documents. It can be represented as ["BM25Retriever"].

                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
                       must be specified explicitly as "QueryClassifier.output_2".
        """
        self.pipeline.add_node(component=component, name=name, inputs=inputs)

    def get_node(self, name: str):
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        component = self.pipeline.get_node(name)
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.pipeline.set_node(name, component)

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        self.pipeline.draw(path)

    def get_nodes_by_class(self, class_type) -> List[Any]:
        """
        Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
        This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
        Example:
        ```python
        from haystack.document_stores.base import BaseDocumentStore
        INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
        res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)
        ```
        :return: List of components that are an instance of the requested class
        """
        return self.pipeline.get_nodes_by_class(class_type)

    def get_document_store(self) -> Optional[BaseDocumentStore]:
        """
        Return the document store object used in the current pipeline.

        :return: Instance of DocumentStore or None
        """
        return self.pipeline.get_document_store()

    def get_type(self) -> str:
        """
        Return the type of the pipeline.

        :return: Type of the pipeline
        """
        return self.pipeline.get_type()

    def eval(
        self,
        labels: List[MultiLabel],
        params: Optional[dict] = None,
        sas_model_name_or_path: Optional[str] = None,
        sas_batch_size: int = 32,
        sas_use_gpu: bool = True,
        add_isolated_node_eval: bool = False,
        custom_document_id_field: Optional[str] = None,
        context_matching_min_length: int = 100,
        context_matching_boost_split_overlaps: bool = True,
        context_matching_threshold: float = 65.0,
    ) -> EvaluationResult:
        """
        Evaluates the pipeline by running the pipeline once per query in debug mode
        and putting together all data that is needed for evaluation, e.g. calculating metrics.

        If you want to calculate SAS (Semantic Answer Similarity) metrics, you have to specify `sas_model_name_or_path`.

        You will be able to control the scope within which an answer or a document is considered correct afterwards (See `document_scope` and `answer_scope` params in `EvaluationResult.calculate_metrics()`).
        Some of these scopes require additional information that already needs to be specified during `eval()`:
        - `custom_document_id_field` param to select a custom document ID from document's meta data for ID matching (only affects 'document_id' scopes)
        - `context_matching_...` param to fine-tune the fuzzy matching mechanism that determines whether some text contexts match each other (only affects 'context' scopes, default values should work most of the time)

        :param labels: The labels to evaluate on
        :param params: Params for the `retriever` and `reader`. For instance,
                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        :param sas_model_name_or_path: SentenceTransformers semantic textual similarity model to be used for sas value calculation,
                                    should be path or string pointing to downloadable models.
        :param sas_batch_size: Number of prediction label pairs to encode at once by CrossEncoder or SentenceTransformer while calculating SAS.
        :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                            Falls back to CPU if no GPU is available.
        :param add_isolated_node_eval: Whether to additionally evaluate the reader based on labels as input instead of output of previous node in pipeline
        :param custom_document_id_field: Custom field name within `Document`'s `meta` which identifies the document and is being used as criterion for matching documents to labels during evaluation.
                                         This is especially useful if you want to match documents on other criteria (e.g. file names) than the default document ids as these could be heavily influenced by preprocessing.
                                         If not set (default) the `Document`'s `id` is being used as criterion for matching documents to labels.
        :param context_matching_min_length: The minimum string length context and candidate need to have in order to be scored.
                           Returns 0.0 otherwise.
        :param context_matching_boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
        :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100]
        """
        output = self.pipeline.eval(
            labels=labels,
            params=params,
            sas_model_name_or_path=sas_model_name_or_path,
            sas_batch_size=sas_batch_size,
            sas_use_gpu=sas_use_gpu,
            add_isolated_node_eval=add_isolated_node_eval,
            custom_document_id_field=custom_document_id_field,
            context_matching_boost_split_overlaps=context_matching_boost_split_overlaps,
            context_matching_min_length=context_matching_min_length,
            context_matching_threshold=context_matching_threshold,
        )
        return output

    def eval_batch(
        self,
        labels: List[MultiLabel],
        params: Optional[dict] = None,
        sas_model_name_or_path: Optional[str] = None,
        sas_batch_size: int = 32,
        sas_use_gpu: bool = True,
        add_isolated_node_eval: bool = False,
        custom_document_id_field: Optional[str] = None,
        context_matching_min_length: int = 100,
        context_matching_boost_split_overlaps: bool = True,
        context_matching_threshold: float = 65.0,
    ) -> EvaluationResult:
        """
         Evaluates the pipeline by running the pipeline once per query in the debug mode
         and putting together all data that is needed for evaluation, for example, calculating metrics.

        To calculate SAS (Semantic Answer Similarity) metrics, specify `sas_model_name_or_path`.

         You can control the scope within which an Answer or a Document is considered correct afterwards (see `document_scope` and `answer_scope` params in `EvaluationResult.calculate_metrics()`).
         For some of these scopes, you need to add the following information during `eval()`:
         - `custom_document_id_field` parameter to select a custom document ID from document's metadata for ID matching (only affects 'document_id' scopes).
         - `context_matching_...` parameter to fine-tune the fuzzy matching mechanism that determines whether text contexts match each other (only affects 'context' scopes, default values should work most of the time).

         :param labels: The labels to evaluate on.
         :param params: Parameters for the `retriever` and `reader`. For instance,
                        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}.
         :param sas_model_name_or_path: Sentence transformers semantic textual similarity model you want to use for the SAS value calculation.
                                     It should be a path or a string pointing to downloadable models.
         :param sas_batch_size: Number of prediction label pairs to encode at once by cross encoder or sentence transformer while calculating SAS.
         :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                             Falls back to CPU if no GPU is available.
         :param add_isolated_node_eval: Whether to additionally evaluate the reader based on labels as input, instead of the output of the previous node in the pipeline.
         :param custom_document_id_field: Custom field name within `Document`'s `meta` which identifies the document and is used as a criterion for matching documents to labels during evaluation.
                                          This is especially useful if you want to match documents on other criteria (for example, file names) than the default document IDs, as these could be heavily influenced by preprocessing.
                                          If not set, the default `Document`'s `id` is used as the criterion for matching documents to labels.
         :param context_matching_min_length: The minimum string length context and candidate need to have to be scored.
                            Returns 0.0 otherwise.
         :param context_matching_boost_split_overlaps: Whether to boost split overlaps (for example, [AB] <-> [BC]) that result from different preprocessing parameters.
                                  If we detect that the score is near a half match and the matching part of the candidate is at its boundaries,
                                  we cut the context on the same side, recalculate the score, and take the mean of both.
                                  Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
         :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100]
        """
        output = self.pipeline.eval_batch(
            labels=labels,
            params=params,
            sas_model_name_or_path=sas_model_name_or_path,
            sas_batch_size=sas_batch_size,
            sas_use_gpu=sas_use_gpu,
            add_isolated_node_eval=add_isolated_node_eval,
            custom_document_id_field=custom_document_id_field,
            context_matching_boost_split_overlaps=context_matching_boost_split_overlaps,
            context_matching_min_length=context_matching_min_length,
            context_matching_threshold=context_matching_threshold,
        )
        return output

    def print_eval_report(
        self,
        eval_result: EvaluationResult,
        n_wrong_examples: int = 3,
        metrics_filter: Optional[Dict[str, List[str]]] = None,
        document_scope: Literal[
            "document_id",
            "context",
            "document_id_and_context",
            "document_id_or_context",
            "answer",
            "document_id_or_answer",
        ] = "document_id_or_answer",
        answer_scope: Literal["any", "context", "document_id", "document_id_and_context"] = "any",
        wrong_examples_fields: Optional[List[str]] = None,
        max_characters_per_field: int = 150,
    ):
        """
        Prints evaluation report containing a metrics funnel and worst queries for further analysis.

        :param eval_result: The evaluation result, can be obtained by running eval().
        :param n_wrong_examples: The number of worst queries to show.
        :param metrics_filter: The metrics to show per node. If None all metrics will be shown.
        :param document_scope: A criterion for deciding whether documents are relevant or not.
            You can select between:
            - 'document_id': Specifies that the document ID must match. You can specify a custom document ID through `pipeline.eval()`'s `custom_document_id_field` param.
                    A typical use case is Document Retrieval.
            - 'context': Specifies that the content of the document must match. Uses fuzzy matching (see `pipeline.eval()`'s `context_matching_...` params).
                    A typical use case is Document-Independent Passage Retrieval.
            - 'document_id_and_context': A Boolean operation specifying that both `'document_id' AND 'context'` must match.
                    A typical use case is Document-Specific Passage Retrieval.
            - 'document_id_or_context': A Boolean operation specifying that either `'document_id' OR 'context'` must match.
                    A typical use case is Document Retrieval having sparse context labels.
            - 'answer': Specifies that the document contents must include the answer. The selected `answer_scope` is enforced automatically.
                    A typical use case is Question Answering.
            - 'document_id_or_answer' (default): A Boolean operation specifying that either `'document_id' OR 'answer'` must match.
                    This is intended to be a proper default value in order to support both main use cases:
                    - Document Retrieval
                    - Question Answering
            The default value is 'document_id_or_answer'.
        :param answer_scope: Specifies the scope in which a matching answer is considered correct.
            You can select between:
            - 'any' (default): Any matching answer is considered correct.
            - 'context': The answer is only considered correct if its context matches as well.
                    Uses fuzzy matching (see `pipeline.eval()`'s `context_matching_...` params).
            - 'document_id': The answer is only considered correct if its document ID matches as well.
                    You can specify a custom document ID through `pipeline.eval()`'s `custom_document_id_field` param.
            - 'document_id_and_context': The answer is only considered correct if its document ID and its context match as well.
            The default value is 'any'.
            In Question Answering, to enforce that the retrieved document is considered correct whenever the answer is correct, set `document_scope` to 'answer' or 'document_id_or_answer'.
        :param wrong_examples_fields: A list of field names to include in the worst samples. By default, "answer", "context", and "document_id" are used.
        :param max_characters_per_field: The maximum number of characters per wrong example to show (per field).
        """
        if wrong_examples_fields is None:
            wrong_examples_fields = ["answer", "context", "document_id"]
        if metrics_filter is None:
            metrics_filter = self.metrics_filter
        self.pipeline.print_eval_report(
            eval_result=eval_result,
            n_wrong_examples=n_wrong_examples,
            metrics_filter=metrics_filter,
            document_scope=document_scope,
            answer_scope=answer_scope,
            wrong_examples_fields=wrong_examples_fields,
            max_characters_per_field=max_characters_per_field,
        )

    def run_batch(self, queries: List[str], params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        Run a batch of queries through the pipeline.

        :param queries: List of query strings.
        :param params: Parameters for the individual nodes of the pipeline. For instance,
                       `params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}`
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default these include the input parameters
                      they received and the output they generated.
                      All debug information can then be found in the dict returned
                      by this method under the key "_debug"
        """
        output = self.pipeline.run_batch(queries=queries, params=params, debug=debug)
        return output


class ExtractiveQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Extractive Question Answering.
    """

    def __init__(self, reader: BaseReader, retriever: BaseRetriever):
        """
        :param reader: Reader instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])
        self.metrics_filter = {"Retriever": ["recall_single_hit"]}

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: The search query string.
        :param params: Params for the `retriever` and `reader`. For instance,
                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default these include the input parameters
                      they received and the output they generated.
                      All debug information can then be found in the dict returned
                      by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class DocumentSearchPipeline(BaseStandardPipeline):
    """
    Pipeline for semantic document search.
    """

    def __init__(self, retriever: BaseRetriever):
        """
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class GenerativeQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Generative Question Answering.
    """

    def __init__(self, generator: BaseGenerator, retriever: BaseRetriever):
        """
        :param generator: Generator instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=generator, name="Generator", inputs=["Retriever"])

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `generator`. For instance,
                       params={"Retriever": {"top_k": 10}, "Generator": {"top_k": 5}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class FAQPipeline(BaseStandardPipeline):
    """
    Pipeline for finding similar FAQs using semantic document search.
    """

    def __init__(self, retriever: BaseRetriever):
        """
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=Docs2Answers(), name="Docs2Answers", inputs=["Retriever"])

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class QuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes documents as input and generates
    questions that it thinks can be answered by the documents.
    """

    def __init__(self, question_generator: QuestionGenerator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Query"])

    def run(self, documents, params: Optional[dict] = None, debug: Optional[bool] = None):
        output = self.pipeline.run(documents=documents, params=params, debug=debug)
        return output

    def run_batch(  # type: ignore
        self,
        documents: Union[List[Document], List[List[Document]]],
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        output = self.pipeline.run_batch(documents=documents, params=params, debug=debug)
        return output


class RetrieverQuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes a query as input, performs retrieval, and then generates
    questions that it thinks can be answered by the retrieved documents.
    """

    def __init__(self, retriever: BaseRetriever, question_generator: QuestionGenerator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Retriever"])

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class QuestionAnswerGenerationPipeline(BaseStandardPipeline):
    """
    This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
    this document, and then performs question answering of this questions using that single document.
    """

    def __init__(self, question_generator: QuestionGenerator, reader: BaseReader):
        setattr(question_generator, "run", self.formatting_wrapper(question_generator.run))
        # Overwrite reader.run function so it can handle a batch of questions being passed on by the QuestionGenerator
        setattr(reader, "run", reader.run_batch)
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["QuestionGenerator"])

    # This is used to format the output of the QuestionGenerator so that its questions are ready to be answered by the reader
    def formatting_wrapper(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            output, output_stream = fn(*args, **kwargs)
            questions = []
            documents = []
            for generated_questions, doc in zip(output["generated_questions"], output["documents"]):
                questions.extend(generated_questions["questions"])
                documents.extend([[doc]] * len(generated_questions["questions"]))
            kwargs["queries"] = questions
            kwargs["documents"] = documents
            return kwargs, output_stream

        return wrapper

    def run(
        self, documents: List[Document], params: Optional[dict] = None, debug: Optional[bool] = None  # type: ignore
    ):
        output = self.pipeline.run(documents=documents, params=params, debug=debug)
        return output


class MostSimilarDocumentsPipeline(BaseStandardPipeline):
    def __init__(self, document_store: BaseDocumentStore):
        """
        Initialize a Pipeline for finding the most similar documents to a given document.
        This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.

        :param document_store: Document Store instance with already stored embeddings.
        """
        # we create a pipeline and add the document store as a node
        # however, we do not want to use the document store's run method,
        # but rather the query_by_embedding method
        # pipeline property is here so the superclass methods that rely on pipeline property work
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Query"])
        self.document_store = document_store

    def run(
        self, document_ids: List[str], filters: Optional[FilterType] = None, top_k: int = 5, index: Optional[str] = None
    ):
        """
        :param document_ids: document ids
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain conditions
        :param top_k: How many documents id to return against single document
        :param index: Optionally specify the name of index to query the document from. If None, the DocumentStore's default index (self.index) will be used.
        """
        self.document_store.return_embedding = True  # type: ignore

        documents = self.document_store.get_documents_by_id(ids=document_ids, index=index)
        query_embs = [doc.embedding for doc in documents]
        similar_documents = self.document_store.query_by_embedding_batch(
            query_embs=query_embs, filters=filters, return_embedding=False, top_k=top_k, index=index  # type: ignore [arg-type]
        )

        self.document_store.return_embedding = False  # type: ignore
        return similar_documents

    def run_batch(  # type: ignore
        self, document_ids: List[str], filters: Optional[FilterType] = None, top_k: int = 5, index: Optional[str] = None
    ):
        """
        :param document_ids: document ids
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain conditions
        :param top_k: How many documents id to return against single document
        :param index: Optionally specify the name of index to query the document from. If None, the DocumentStore's default index (self.index) will be used.
        """
        return self.run(document_ids=document_ids, filters=filters, top_k=top_k, index=index)


class TextIndexingPipeline(BaseStandardPipeline):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        text_converter: Optional[TextConverter] = None,
        preprocessor: Optional[PreProcessor] = None,
    ):
        """
        Initialize a basic Pipeline that converts text files into Documents and indexes them into a DocumentStore.

        :param document_store: The DocumentStore to index the Documents into.
        :param text_converter: A TextConverter object to be used in this pipeline for converting the text files into Documents.
        :param preprocessor: A PreProcessor object to be used in this pipeline for preprocessing Documents.
        """

        self.pipeline = Pipeline()
        self.document_store = document_store
        self.text_converter = text_converter or TextConverter()
        self.preprocessor = preprocessor or PreProcessor()
        self.pipeline.add_node(component=self.text_converter, name="TextConverter", inputs=["File"])
        self.pipeline.add_node(component=self.preprocessor, name="PreProcessor", inputs=["TextConverter"])
        self.pipeline.add_node(component=self.document_store, name="DocumentStore", inputs=["PreProcessor"])

    def run(self, file_path):
        return self.pipeline.run(file_paths=[file_path])

    def run_batch(self, file_paths):
        return self.pipeline.run_batch(file_paths=file_paths)
