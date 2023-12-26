import hashlib
import io
import json
import math
import os
#import semantra
import psycopg2
import click
import numpy as np
import pkg_resources
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, request, send_file, send_from_directory, url_for, render_template, \
    redirect, flash, Blueprint, session
from tqdm import tqdm

from models import BaseModel, as_numpy, models, TransformerModel
from util import (
    HASH_LENGTH,
    file_md5,
    get_annoy_filename,
    get_config_filename,
    get_embeddings_filename,
    get_num_annoy_embeddings,
    get_num_embeddings,
    get_offsets,
    get_tokens_filename,
    join_text_chunks,
    load_annoy_db,
    read_embeddings_file,
    sort_results,
    write_annoy_db,
    write_embedding,
)
import searchIKDocs
from collections import defaultdict



#VERSION = pkg_resources.require("semantra")[0].version # REVIEW IF NEEDED
DEFAULT_ENCODING = "utf-8"
DEFAULT_PORT = 10000
package_directory = os.path.dirname(os.path.abspath(__file__))
selected_folder_name = 'judgements'

class Content:
    def __init__(self, rawtext, filename):
        self.rawtext = rawtext
        self.filename = filename
        self.filetype = "text"


def get_text_content(md5, filename, semantra_dir, force, silent, encoding):
    if filename.endswith(".pdf"):
        return get_pdf_content(md5, filename, semantra_dir, force, silent)

    with open(filename, "r", encoding=encoding, errors="ignore") as f:
        rawtext = f.read()
        return Content(rawtext, filename)


TRANSFORMER_POOL_DEFAULT = 15000



class Document:
    def __init__(
        self,
        filename,
        md5,
        semantra_dir,
        base_filename,
        config,
        embeddings_filenames,
        use_annoy,
        annoy_filenames,
        windows,
        offsets,
        tokens_filename,
        num_dimensions,
        encoding,
    ):
        self.filename = filename
        self.md5 = md5
        self.semantra_dir = semantra_dir
        self.base_filename = base_filename
        self.config = config
        self.embeddings_filenames = embeddings_filenames
        self.use_annoy = use_annoy
        self.annoy_filenames = annoy_filenames
        self.windows = windows
        self.offsets = offsets
        self.tokens_filename = tokens_filename
        self.num_dimensions = num_dimensions
        self.encoding = encoding

    @property
    def content(self):
        return get_text_content(
            self.md5, self.filename, self.semantra_dir, False, True, self.encoding
        )

    @property
    def text_chunks(self):
        with open(self.tokens_filename, "r") as f:
            return json.loads(f.read())

    @property
    def num_embeddings(self):
        return len(self.offsets[0])

    @property
    def embedding_db(self):
        if not self.use_annoy:
            raise ValueError("Embeddings are not stored in Annoy database")
        return load_annoy_db(self.annoy_filenames[0], self.num_dimensions)

    @property
    def embeddings(self):
        results, embedding_count = read_embeddings_file(
            self.embeddings_filenames[0],
            self.num_dimensions,
            self.num_embeddings,
        )
        assert embedding_count == self.num_embeddings
        return results


def process(
    filename,
    semantra_dir,
    model,
    num_dimensions,
    use_annoy,
    num_annoy_trees,
    windows,
    cost_per_token,
    pool_count,
    pool_size,
    force,
    silent,
    no_confirm,
    encoding,
):
    # Check if semantra dir exists
    if not os.path.exists(semantra_dir):
        os.makedirs(semantra_dir)

    # Get the md5 and config
    md5 = file_md5(filename)
    base_filename = os.path.basename(filename)
    config = model.get_config()
    if encoding != DEFAULT_ENCODING:
        config["encoding"] = encoding
    config_hash = hashlib.shake_256(json.dumps(config).encode()).hexdigest(HASH_LENGTH)

    # File names
    tokens_filename = os.path.join(semantra_dir, get_tokens_filename(md5, config_hash))
    config_filename = os.path.join(semantra_dir, get_config_filename(md5, config_hash))
    
    should_calculate_tokens = True
    if force or not os.path.exists(tokens_filename):
        # Calculate tokens to get text chunks
        content = get_text_content(md5, filename, semantra_dir, force, silent, encoding)
        text = content.rawtext
        tokens = model.get_tokens(text)
        should_calculate_tokens = False
        text_chunks = model.get_text_chunks(text, tokens)
        with open(tokens_filename, "w") as f:
            f.write(json.dumps(text_chunks))
    else:
        with open(tokens_filename, "r") as f:
            text_chunks = json.loads(f.read())
    num_tokens = len(text_chunks)

    # Get embedding offsets based on config parameters
    (
        offsets,
        num_embedding_tokens,
    ) = get_offsets(num_tokens, windows)

    # Full config contains additional details
    full_config = {
        **config,
        "filename": filename,
        "md5": md5,
        "base_filename": base_filename,
        "num_dimensions": num_dimensions,
        "cost_per_token": cost_per_token,
        "windows": windows,
        "num_tokens": num_tokens,
        "num_embeddings": len(offsets),
        "num_embedding_tokens": num_embedding_tokens,
        "use_annoy": use_annoy,
        "num_annoy_trees": num_annoy_trees,
        "semantra_version": "0.1.7",
    }
    '''
    if force or not os.path.exists(config_filename):
        if cost_per_token is not None and not no_confirm:
            click.confirm(
                f"Tokens will cost ${num_embedding_tokens * cost_per_token:.2f}. Proceed?",
                abort=True,
            )

    '''

    # Write out the config every time
    with open(config_filename, "w") as f:
        f.write(json.dumps(full_config))

    embeddings_filenames = []
    annoy_filenames = []
    with tqdm(
        total=num_embedding_tokens,
        desc="Calculating embeddings",
        leave=False,
        disable=silent,
    ) as pbar:
        for (size, offset, rewind), sub_offsets in zip(windows, offsets):
            embeddings_filename = os.path.join(
                semantra_dir,
                get_embeddings_filename(md5, config_hash, size, offset, rewind),
            )
            annoy_filename = os.path.join(
                semantra_dir,
                get_annoy_filename(
                    md5, config_hash, size, offset, rewind, num_annoy_trees
                ),
            )
            embeddings_filenames.append(embeddings_filename)
            annoy_filenames.append(annoy_filename)

            if os.path.exists(embeddings_filename) and (
                not use_annoy or os.path.exists(annoy_filename)
            ):
                num_embeddings = get_num_embeddings(embeddings_filename, num_dimensions)
                if use_annoy:
                    num_annoy_embeddings = get_num_annoy_embeddings(
                        annoy_filename, num_dimensions
                    )

                if (
                    not force
                    and num_embeddings == len(sub_offsets)
                    and (not use_annoy or num_annoy_embeddings == len(sub_offsets))
                ):
                    # Embedding is fully calculated
                    continue

            if should_calculate_tokens:
                tokens = model.get_tokens(join_text_chunks(text_chunks))
                should_calculate_tokens = False

            # Read embeddings if they exist
            embedding_index = 0
            if not force and os.path.exists(embeddings_filename):
                embeddings, embedding_index = read_embeddings_file(
                    embeddings_filename, num_dimensions, len(sub_offsets)
                )
            else:
                embeddings = np.empty(
                    (len(sub_offsets), num_dimensions), dtype=np.float32
                )
                embedding_index = 0

            num_skip = embedding_index
            iteration = 0

            # Write embeddings
            pool = []
            pool_token_count = 0

            with open(embeddings_filename, "ab") as f:

                def flush_pool():
                    nonlocal pool, pool_token_count, embeddings, embedding_index, f

                    if len(pool) > 0:
                        embedding_results = model.embed(tokens, pool)
                        # Call .cpu if embedding_results contains it
                        if hasattr(embedding_results, "cpu"):
                            embedding_results = embedding_results.cpu()
                        embeddings[
                            embedding_index : embedding_index + len(pool)
                        ] = embedding_results
                        for embedding in embedding_results:
                            write_embedding(f, embedding, num_dimensions)
                        embedding_index += len(pool)
                        pool = []
                        pool_token_count = 0

                for offset in sub_offsets:
                    size = offset[1] - offset[0]

                    # Skip if already calculated
                    if iteration < num_skip:
                        iteration += 1
                        pbar.update(size)
                        continue

                    window_text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                    if len(window_text) == 0:
                        pbar.update(size)
                        continue

                    pool.append(offset)
                    pool_token_count += size
                    if (
                        pool_count is not None and len(pool) >= pool_count
                    ) or pool_token_count >= pool_size:
                        flush_pool()
                    pbar.update(size)

                flush_pool()

            # Write embeddings db
            if use_annoy:
                write_annoy_db(
                    filename=annoy_filename,
                    num_dimensions=num_dimensions,
                    embeddings=embeddings,
                    num_trees=num_annoy_trees,
                )

    return Document(
        filename=filename,
        md5=md5,
        semantra_dir=semantra_dir,
        base_filename=base_filename,
        config=full_config,
        embeddings_filenames=embeddings_filenames,
        use_annoy=use_annoy,
        annoy_filenames=annoy_filenames,
        windows=windows,
        offsets=offsets,
        tokens_filename=tokens_filename,
        num_dimensions=num_dimensions,
        encoding=encoding,
    )


def process_windows(windows: str) -> "list[tuple[int, int, int]]":

    if not isinstance(windows, str):
        windows = str(windows)

    yield 128, 0, 16
    


def main(
    windows="128_0_16",
    no_server=False,
    port=10000,
    host="0.0.0.0",
    pool_size=None,
    pool_count=None,
    doc_token_pre=None,
    doc_token_post=None,
    query_token_pre=None,
    query_token_post=None,
    model="openai",
    transformer_model=None,
    encoding=DEFAULT_ENCODING,
    num_annoy_trees=100,
    num_results=30,
    annoy=True,
    svm=False,
    svm_c=1.0,
    explain_split_count=9,
    explain_split_divide=6,
    num_explain_highlights=2,
    force=False,
    silent=False,
    no_confirm=False,
    version=False,
    list_models=False,
    show_semantra_dir=False,
    semantra_dir=None,  # auto
):

    folder_path = "text-documents"
    
    def get_filenames():

        judgement_files = []
        books_files = []
        preview_files = []
   
        custom_judgment_files = []
        

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if root.endswith("books"):
                    books_files.append(file_path)
                elif root.endswith("preview"):
                    preview_files.append(file_path)
                elif root.endswith("judgments"):
                    judgement_files.append(file_path)
                elif root.endswith(f"custom_judgments_IK"):
                    custom_judgment_files.append(file_path)

        # Print the lists of file names
        print("Judgement Files:", judgement_files)
        print("Books Files:", books_files)
        print("Preview Files:", preview_files)
        print("Custom_judgment_files:", custom_judgment_files)

        return judgement_files, books_files, preview_files, custom_judgment_files

    judgment_files, books_files, preview_files, custom_judgment_files = get_filenames()

    filename = judgment_files
    
    if version:
        print(version)
        return

    

    if semantra_dir is None:
        semantra_dir = "./processed"
    
    # Load environment from Semantra dir
    env_path = os.path.join(semantra_dir, ".env")
    load_dotenv(env_path)

    processed_windows = list(process_windows(windows))

    
    if transformer_model is not None:
        # Handle custom transformers model
        if pool_size is None:
            pool_size = TRANSFORMER_POOL_DEFAULT

        cost_per_token = None
        model = TransformerModel(
            transformer_model
        )
    else:
        # Pull preset model
        model_config = models[model]
        cost_per_token = model_config["cost_per_token"]
        if pool_size is None:
            pool_size = model_config["pool_size"]
        if pool_count is None:
            pool_count = model_config.get("pool_count", None)
        model: BaseModel = model_config["get_model"]()

    # Check if model is compatible
    if svm and model.is_asymmetric():
        raise ValueError(
            "SVM is not compatible with asymmetric models. "
            "Please use a symmetric model or kNN."
        )
    print ("Filename is", filename) #FOR DEBUGGING
    documents = {}

    judgements_documents = {}
    books_documents = {}
    preview_documents = {}
    custom_judgment_documents = {}
        
    def process_custom(custom_judgment_files):
        pbar = tqdm(custom_judgment_files, disable=silent)
        print ("pbar is", pbar)

        for fn in pbar:
            pbar.set_description(f"{os.path.basename(fn)}")
            print ("fn is", fn) #FOR DEBUGGING
            custom_judgment_documents[fn] = process(
                filename=fn,
                semantra_dir=semantra_dir,
                model=model,
                num_dimensions=model.get_num_dimensions(),
                use_annoy=annoy,
                num_annoy_trees=num_annoy_trees,
                windows=processed_windows,
                cost_per_token=cost_per_token,
                pool_count=pool_count,
                pool_size=pool_size,
                force=force,
                silent=silent,
                no_confirm=no_confirm,
                encoding=encoding,
            )
        return custom_judgment_documents
    

    pbar = tqdm(preview_files, disable=silent)
    print ("pbar is", pbar) # FOR DEBUGGING
    for fn in pbar:
        pbar.set_description(f"{os.path.basename(fn)}")
        print ("fn is", fn) #FOR DEBUGGING
        preview_documents[fn] = process(
            filename=fn,
            semantra_dir=semantra_dir,
            model=model,
            num_dimensions=model.get_num_dimensions(),
            use_annoy=annoy,
            num_annoy_trees=num_annoy_trees,
            windows=processed_windows,
            cost_per_token=cost_per_token,
            pool_count=pool_count,
            pool_size=pool_size,
            force=force,
            silent=silent,
            no_confirm=no_confirm,
            encoding=encoding,
        )

    
        
    

        
    documents = preview_documents
    cached_content = None
    cached_content_filename = None

    def get_content(filename):
        print("that is called ........................")
        nonlocal cached_content, cached_content_filename
        # Check if we can pull from cache
        if filename == cached_content_filename:
            return cached_content
        # If not, grab content
        content = documents[filename].content
        # Cache the content
        cached_content_filename = filename
        cached_content = content
        # Return the now-cached content
        return content

    
    # Start a Flask server
    app = Flask(__name__)
    
    current_directory = os.getcwd()
    

    @app.route("/", methods=['GET', 'POST'])
    def base():
        nonlocal documents
        selected_folder_name = "preview"
        bool_searched_value = session.get('bool_searched_value')
        print("Boolean value is", bool_searched_value)
        number_of_documents = 2
        if request.method=='POST':
            folder_name = request.form.get('folder_name')
            selected_folder_name = folder_name
            print ("Selected folder is", folder_name)
            def documentsetter(folder_name, bool_searched_value):
                if folder_name == 'custom_judgments':
                    folder_path=f'./text-documents/custom_judgments_IK'
                    try:
                        # List all files in the folder
                        custom_judgment_files = get_filenames()[3]
                        number_of_documents = len(custom_judgment_files)
                        print("CustomJ Files preloaded are:", custom_judgment_files)
                        documents = process_custom(custom_judgment_files)
                    except Exception as e:
                        print(f"Error: {e}")
                        number_of_documents = 0
                        documents = None

                    
                        
                else:
                    documents = preview_documents
                    number_of_documents = 2

                return documents, number_of_documents, selected_folder_name
                    

            documents, number_of_documents, selected_folder_name = documentsetter(folder_name, bool_searched_value)
        
        if number_of_documents != 0:
            return render_template( "index.html", selected_folder_name=selected_folder_name)
        else:
            return redirect(url_for('searchIK'))
                        
            
        
    @app.route("/selected", methods=['GET', 'POST'])
    def selected_base():
        nonlocal documents
        custom_judgment_files = get_filenames()[3]
        number_of_documents = len(custom_judgment_files)
        print("CustomJ Files are:", custom_judgment_files)
        documents = process_custom(custom_judgment_files)
        print("custom_judgment_documents are:" , documents)
        
        return render_template("index.html", selected_folder_name="custom_judgments")

    @app.route("/custom-judgments", methods=['POST', 'GET'])
    def searchIK():
        documentlist = defaultdict(list)
        '''This module utilises the IndianKanoon API
        You must register for your own key - https://api.indiankanoon.org'''


        
        if request.method == 'POST' and request.form.get('name') == 'searchquery':
            documentlist = searchIKDocs.get_titles(request.form.get('searchquery'))
        
            
        return render_template("searchpage.html", documentlist=documentlist)
        
    @app.route('/process_selected_documents', methods=['POST'])
    def process_selected_documents():

        if request.method == 'POST' and request.form.get('name') == 'searchdocs':
            searchdocs = request.form.get('searchdocs')
            documenttexts = searchIKDocs.get_documents_from_list(searchdocs)

        elif request.method == 'POST' and request.form.get('name') != 'searchdocs':
            selected_documents = request.form.getlist('selected_documents')
            print(selected_documents)
            documenttexts = searchIKDocs.get_documents(selected_documents)
        os.chdir(f'./text-documents/custom_judgments_IK')
        for index, value in enumerate(documenttexts):
            with open (f"Document No. {index}", "w") as file:
                file.write(value)        
        os.chdir(current_directory)
        
        return redirect(url_for('selected_base'))

    @app.route('/clear')
    def clear():
        folder_path=f'./text-documents/custom_judgments_IK'
        print (folder_path)
        def remove_all_files(folder_path):
            try:
                # Get the list of files in the folder
                files = os.listdir(folder_path)

                print(files)

                # Iterate through each file and remove it
                for file_name in files:
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Removed file: {file_path}")

                print(f"All files removed from {folder_path}")

            except Exception as e:
                print(f"Error: {e}")

        remove_all_files(folder_path)

        return redirect(url_for('base'))

    @app.route("/preview", methods=['GET', 'POST'])
    def preview_base():
        nonlocal documents
        selected_folder_name = "preview"
        documents = preview_documents
        return render_template( "preview-index.html", selected_folder_name=selected_folder_name)


    @app.route("/<path:path>")
    def home(path):
        return send_from_directory("client_public", path)
    
    

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
    
    @app.route("/api/files/", methods=["GET"])
    def files():
        nonlocal documents
        return jsonify(
            [
                {
                    "basename": doc.base_filename,
                    "filename": doc.filename,
                    "filetype": doc.content.filetype,
                }
                for doc in documents.values()
            ]
        )



    @app.route("/api/query", methods=["POST"])
    def query():
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        if svm:
            return querysvm()
        if annoy:
            return queryann()

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            # Get kNN with cosine similarity
            distances = np.dot(embeddings, embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
            )
            sorted_ix = np.argsort(-distances)

            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = float(distances[index])
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(sort_results(results, True))

    @app.route("/api/querysvm", methods=["POST"])
    def querysvm():
        from sklearn import svm

        queries = request.json["queries"]
        preferences = request.json["preferences"]

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)
        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            x = np.concatenate([embeddings, embedding[None, ...]])
            y = np.zeros(len(embeddings) + 1)
            y[-1] = 1

            # Train the svm
            clf = svm.LinearSVC(
                class_weight="balanced",
                verbose=False,
                max_iter=10000,
                tol=1e-6,
                C=svm_c,
            )
            clf.fit(x, y)

            # Infer similarities
            similarities = clf.decision_function(x)[: len(embeddings)]
            sorted_ix = np.argsort(-similarities)

            text_chunks = doc.text_chunks
            offsets = doc.offsets
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = similarities[index]
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])

        return jsonify(sort_results(results, True))

    @app.route("/api/queryann", methods=["POST"])
    def queryann():
        queries = request.json["queries"]
        preferences = request.json["preferences"]

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embedding_db = doc.embedding_db
            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for [index, distance] in zip(
                *embedding_db.get_nns_by_vector(embedding, num_results, -1, True)
            ):
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        # Convert distance from Euclidean distance of normalized vectors to cosine
                        "distance": 1 - distance**2.0 / 2.0,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(sort_results(results, True))

    @app.route("/api/explain", methods=["POST"])
    def explain():
        filename = request.json["filename"]
        offset = request.json["offset"]
        tokens = documents[filename].text_chunks[offset[0] : offset[1]]
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        # Find hot-spots within the result tokens
        def get_splits(divide_factor=2, num_splits=3, start=0, end=len(tokens)):
            window_length = math.ceil((end - start) / divide_factor)
            split_length = math.ceil((end - start) / num_splits)
            splits = []
            for i in range(num_splits):
                splits.append(
                    (
                        start + i * split_length,
                        min(end, start + i * split_length + window_length),
                    )
                )
            return splits

        def exclude_window(start, end):
            nonlocal tokens
            return join_text_chunks(tokens[:start] + tokens[end:])

        def get_highest_ranked_split(splits):
            nonlocal tokens, embedding
            split_queries = [exclude_window(start, end) for start, end in splits]
            split_windows = np.array(
                [
                    as_numpy(model.embed_document(split_query))
                    for split_query in split_queries
                ]
            )
            distances = split_windows.dot(embedding) / (
                np.linalg.norm(split_windows, axis=1) * np.linalg.norm(embedding)
            )
            # Return the splits in order of highest to lowest ranked
            return sorted(zip(splits, distances), key=lambda x: x[1], reverse=False)

        def as_tokens(splits):
            nonlocal tokens
            indices = sorted([split[0] for split in splits], key=lambda x: x[0])
            last_index = 0
            chunks = []

            def append(start, end, type):
                if start >= end:
                    return
                nonlocal chunks, tokens
                chunks.append(
                    {
                        "text": join_text_chunks(tokens[start:end]),
                        "type": type,
                    }
                )

            for index in indices:
                append(last_index, index[0], "normal")
                append(max(index[0], last_index), index[1], "highlight")
                last_index = index[1]

            append(last_index, len(tokens), "normal")
            return chunks

        splits = get_splits(
            divide_factor=explain_split_divide,
            num_splits=explain_split_count,
            start=0,
            end=len(tokens),
        )
        top_splits = get_highest_ranked_split(splits)[:num_explain_highlights]
        return jsonify(as_tokens(top_splits))

    @app.route("/api/getfile", methods=["GET"])
    def getfile():
        filename = request.args.get("filename")
        content = get_content(filename)
        filename = content.filename
        return send_file(filename)

    @app.route("/api/pdfpositions", methods=["GET"])
    def pdfpositions():
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype == "pdf":
            return jsonify(content.positions)
        else:
            return jsonify([])

    @app.route("/api/pdfpage", methods=["GET"])
    def pdfpage():
        filename = request.args.get("filename")
        content = get_content(filename)
        page = request.args.get("page")
        scale = request.args.get("scale")
        if content.filetype == "pdf":
            pil_image = content.get_page_image_pil(int(page), float(scale))
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            response = make_response(img_byte_arr.getvalue())
            response.headers.set("Content-Type", "image/png")
            return response

    @app.route("/api/pdfchars", methods=["GET"])
    def pdfchars():
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype != "pdf":
            return jsonify([])
        page = request.args.get("page")
        return jsonify(content.get_page_chars(int(page)))

    @app.route("/api/text", methods=["GET"])
    def text():
        global selected_folder_name
        filename = request.args.get("filename")

        return jsonify(documents[filename].text_chunks)

    app.run(host='127.0.0.1', port=5000, debug=True)




if __name__ == "__main__":
    folder_name = "./text-documents/custom_judgments_IK"  # Change this to the desired folder name

    # Check if the folder exists
    if not os.path.exists(folder_name):
        # If it doesn't exist, create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    main()
    
