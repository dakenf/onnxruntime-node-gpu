{
  "targets": [
    {
      "target_name": "onnxruntine_node_gpu",
      "sources": ["src/main.cpp"],
      "include_dirs": ["<!@(node -p \"require('node-addon-api').include\")"],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "libraries": []
    }
  ]
}
