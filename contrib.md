# Contributing to Hugot

## Code architecture

Hugot is organised as follows.

The main entrypoints to create and run pipelines are the Session struct and the NewPipeline/GetPipeline functions in hugot.go. 

The Session struct is created with the NewSession method and holds created pipelines, where each pipeline is assigned a name alias.

The NewPipeline function creates new pipelines and stores them in the Session struct (note: it's not a struct method but a stand-alone function, 
since it uses generics and generics are not allowed in struct methods). It
takes a session object and a pipeline configuration struct as input, and updates the session object with the new pipeline, returning it.

Finally, the GetPipeline function allows one to retrieve a specific pipeline from the session based on its alias.

The basic abstractions are contained in the pipelines/pipeline.go file. This file defines:

- the BasePipeline struct, which holds the basic attributes and general methods of any pipeline, such as e.g. ModelPath (every pipeline must have a path to a model), and the loadModel() struct method to load an onnx model. This struct can be embedded by specific pipelines.
- The pipeline interface. Every implemented pipeline should implement this interface.

A specific pipeline implementation, e.g. TokenClassificationPipeline (see pipelines/tokenClassification.go), can then embed the BasePipeline struct to implement the basic attributes and struct methods of a pipeline. It should then implement the pipeline interface. 

New pipelines can be implemented in their own file inside the pipelines folder. 

1. Create a pipeline struct for your new pipeline (e.g. see the TokenClassificationPipeline struct)
2. Embed the BasePipeline struct so you have the base methods and attributes
3. Implement the pipeline interface methods (overriding the basePipeline methods if needed)
4. Update the Session and NewPipeline/GetPipeline functions in hugot.go to be able to create pipelines of your type

## Contribution process

1. create or find an issue for your contribution
2. fork and develop
3. add tests and make sure the full test suite passes and test coverage does not dip below 80%
4. create a MR linking to the relevant issue

Thank you for contributing to hugot!

## Development environment

The easiest way to contribute to hugot is by developing inside a docker container that has both the tokenizer and onnxruntime libraries.
From the source folder, it should be as easy as:

```bash
make start-dev-container
```

this will download the test models, build the test container, and launch it (see [compose-dev](./compose-dev.yaml)), mounting the source code at /home/testuser/repositories/hugot. Then you can attach to the container with e.g. vscode remote extension as testuser. The vscode attached container configuration file can be set to:

```
{
    "remoteUser": "testuser",
    "workspaceFolder": "/home/testuser/repositories/hugot",
    "extensions": [
		"bierner.markdown-preview-github-styles",
		"golang.go",
		"ms-azuretools.vscode-docker"
	],
    "remoteEnv": {"GOPATH": "/home/testuser/go"}
}
```

Once you're done, you can tear the container down with:

```bash
make stop-dev-container
```

Alternatively, you can use your IDE devcontainer support, and point it to the [Dockerfile](./Dockerfile).

If you prefer to develop on bare metal, you will need to download the tokenizers.a to /usr/lib/tokenizers.a and onnxruntime.so to /usr/lib/onnxruntime.so.

## Run the tests

The full test suite can be run as follows. From the source folder:

```bash
make clean run-tests
```

This will build a test image and run all tests in a container. A testTarget folder will appear in the source directory with the test results.
