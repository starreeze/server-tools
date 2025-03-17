# Environment quick init

## Server bashrc:

```shell
mvdir () {
        cp -rln $1 $2
        rm -rf $1
}
gpu () {
        export CUDA_VISIBLE_DEVICES=$1
}
tf () {
        export CUDA_HOME=/home/data_91_c/cuda_tools/cuda-11.2
        export CUDNN_HOME=/home/data_91_c/cuda_tools/cudnn-8.1.0-cuda-11.2
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$LD_LIBRARY_PATH
        conda activate /home/data_91_c/xsy/tf29py38
}
torch20 () {
        conda activate /home/data_91_c/xsy/torch20py310
}
torch23 () {
        conda activate /home/data_91_c/xsy/torch23py311
}
slp () {
        export http_proxy=$PROXY
        export https_proxy=$PROXY
        export ftp_proxy=$PROXY
        export all_proxy=$PROXY
}
export PROXY=''
# 开启系统代理
proxy_on() {
        export http_proxy=http://127.0.0.1:7890
        export https_proxy=http://127.0.0.1:7890
        export no_proxy=127.0.0.1,localhost
        export HTTP_PROXY=http://127.0.0.1:7890
        export HTTPS_PROXY=http://127.0.0.1:7890
        export NO_PROXY=127.0.0.1,localhost
        echo -e "\033[32m[√] 已开启代理\033[0m"
}

# 关闭系统代理
proxy_off(){
        unset http_proxy
        unset https_proxy
        unset no_proxy
        unset HTTP_PROXY
        unset HTTPS_PROXY
        unset NO_PROXY
        echo -e "\033[31m[×] 已关闭代理\033[0m"
}

export HF_ENDPOINT=https://hf-mirror.com

script_dir_path='/home/data_91_c/xsy/scripts'

alias torch=torch20
alias talloc="python $script_dir_path/allocate.py"

export SOFTWARE_BASE=/tmp/software
export PATH=$SOFTWARE_BASE/bin:$SOFTWARE_BASE/sbin:/home/xingsy/.local/bin:/home/nfs04/xingsy/bin:$PATH
LD_BASE=$SOFTWARE_BASE/usr/lib
export LD_LIBRARY_PATH=$( find $LD_BASE -type d -printf "%p:" )$LD_LIBRARY_PATH

alias pipi="python -m pip install -i https://mirrors.nju.edu.cn/pypi/web/simple/ --proxy ''"

export CUDA_HOME=/home/nfs04/cuda_tools/cuda-12.1
export CUDA_VISIBLE_DEVICES=-1
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=300

echo -e "\033[32mX-script collection initialized successfully. Now setting up conda ...\033[0m"

# conda here
```

## Windows profile:

```powershell
function py {
    D:\software\Python310\python.exe @args
}

function python {
    D:\software\Python310\python.exe @args
}

function pipi {
    pip install --index-url https://mirrors.nju.edu.cn/pypi/web/simple @args
}

function testc {
    py D:\OneDrive\code\python\clash\test_speed.py @args
}

function fixc {
    py D:\OneDrive\code\python\clash\fix.py @args
}

function njup {
    docker run --rm --device /dev/net/tun --cap-add NET_ADMIN -ti -e PASSWORD=-10123 -e URLWIN=1 -v $HOME/.ecdata:/root -p 127.0.0.1:5901:5901 -p 127.0.0.1:1080:1080 -p 127.0.0.1:8888:8888 hagb/docker-easyconnect:7.6.7
}

function ssh-proxy {
    cp $HOME/.ssh/config.proxy $HOME/.ssh/config
}

function ssh-direct {
    cp $HOME/.ssh/config.direct $HOME/.ssh/config
}

function recv {
    cd D:\OneDrive\code\python\diffuse-server
    py files.py -d R: @args
}

function slp {
    param (
        [string]$ProxyAddress = "127.0.0.1:7890"
    )

    # Set environment variables for the current session
    $env:http_proxy = "http://$ProxyAddress"
    $env:https_proxy = "http://$ProxyAddress"

    Write-Host "Proxy set to $ProxyAddress for this terminal session."
}

function usp {
    Remove-Item Env:http_proxy -ErrorAction SilentlyContinue
    Remove-Item Env:https_proxy -ErrorAction SilentlyContinue

    Write-Host "Proxy has been removed for this terminal session."
}

function reset-cursor {
    py D:\OneDrive\code\scripts\reset_cursor.py
}

function aria2 {
    aria2c --enable-rpc --rpc-listen-all --rpc-allow-origin-all @args
}
```

## VSCode config

```json
{
  "editor.suggestSelection": "first",
  "vsintellicode.modify.editor.suggestSelection": "automaticallyOverrodeDefaultValue",
  "editor.fontSize": 17,
  "leetcode.endpoint": "leetcode-cn",
  "markdown.preview.fontSize": 18,
  "leetcode.hint.configWebviewMarkdown": false,
  "leetcode.hint.commentDescription": false,
  "leetcode.defaultLanguage": "cpp",
  "C_Cpp.updateChannel": "Insiders",
  "editor.formatOnPaste": true,
  "editor.formatOnSave": true,
  "editor.formatOnType": true,
  "editor.fontFamily": "'DejaVu Sans Mono'",
  "python.languageServer": "Pylance",
  "python.pythonPath": "/usr/bin/python3",
  "python.analysis.completeFunctionParens": true,
  "editor.wordWrap": "on",
  "leetcode.hint.commandShortcut": false,
  "workbench.editorAssociations": {
    "{hexdiff}:/**/*.*": "hexEditor.hexedit",
    "*.ipynb": "jupyter-notebook",
    "*.png": "imagePreview.previewEditor",
    "*.md": "default",
    "*.wav": "vscode.audioPreview",
    "{git,gitlens,git-graph}:/**/*.{md,csv,svg}": "default",
    "*.pdf": "latex-workshop-pdf-hook"
  },
  "git.enableSmartCommit": true,
  "explorer.confirmDelete": false,
  "editor.tabCompletion": "on",
  "editor.stickyTabStops": true,
  "todo-tree.tree.showScanModeButton": false,
  "diffEditor.renderSideBySide": false,
  "terminal.integrated.defaultProfile.linux": "fish",
  "python.autoComplete.addBrackets": true,
  "python.diagnostics.sourceMapsEnabled": true,
  "explorer.confirmDragAndDrop": false,
  "notebook.cellToolbarLocation": {
    "default": "right",
    "jupyter-notebook": "left"
  },
  "C_Cpp.autocompleteAddParentheses": true,
  "C_Cpp.default.cppStandard": "c++20",
  "C_Cpp.default.cStandard": "c17",
  "C_Cpp.default.intelliSenseMode": "windows-gcc-x64",
  "C_Cpp.experimentalFeatures": "Enabled",
  "C_Cpp.intelliSenseEngineFallback": "Enabled",
  "C_Cpp.formatting": "vcFormat",
  "C_Cpp.vcFormat.indent.preserveComments": true,
  "C_Cpp.vcFormat.newLine.beforeOpenBrace.block": "sameLine",
  "C_Cpp.vcFormat.newLine.beforeOpenBrace.function": "sameLine",
  "C_Cpp.vcFormat.newLine.beforeOpenBrace.lambda": "sameLine",
  "C_Cpp.vcFormat.newLine.beforeOpenBrace.namespace": "sameLine",
  "C_Cpp.vcFormat.newLine.beforeOpenBrace.type": "sameLine",
  "C_Cpp.vcFormat.newLine.closeBraceSameLine.emptyType": true,
  "C_Cpp.vcFormat.newLine.closeBraceSameLine.emptyFunction": true,
  "C_Cpp.workspaceSymbols": "All",
  "C_Cpp.workspaceParsingPriority": "high",
  "C_Cpp.vcFormat.indent.lambdaBracesWhenParameter": false,
  "C_Cpp.vcFormat.indent.multiLineRelativeTo": "statementBegin",
  "python.formatting.provider": "none",
  "git.confirmSync": false,
  "back-n-forth.withLastEditLocation": true,
  "background.useFront": false,
  "c-cpp-flylint.clang.standard": ["c++20"],
  "c-cpp-flylint.flexelint.enable": false,
  "c-cpp-flylint.lizard.enable": false,
  "c-cpp-flylint.clang.enable": false,
  "cmake.configureOnOpen": true,
  "editor.unicodeHighlight.nonBasicASCII": false,
  "files.exclude": {
    "**/.classpath": true,
    "**/.project": true,
    "**/.settings": true,
    "**/.factorypath": true
  },
  "[json]": {
    "editor.defaultFormatter": "vscode.json-language-features"
  },
  "leetcode.useEndpointTranslation": false,
  "leetcode-cpp-debugger.source": "[online]leetcode-cn.com",
  "leetcode-cpp-debugger.deleteTemporaryContents": false,
  "redhat.telemetry.enabled": true,
  "security.workspace.trust.untrustedFiles": "open",
  "editor.inlineSuggest.enabled": true,
  "[jsonc]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "cmake.configureEnvironment": {
    "CMAKE_PREFIX_PATH": "/home/xsy/Qt/5.15.2/gcc_6"
  },
  "cmake.environment": {
    "CMAKE_PREFIX_PATH": "/home/xsy/Qt/5.15.2/gcc_64"
  },
  "terminal.integrated.enableMultiLinePasteWarning": false,
  "terminal.integrated.inheritEnv": false,
  "workbench.colorTheme": "Default Light+",
  "editor.unicodeHighlight.invisibleCharacters": false,
  "[javascript]": {
    "editor.defaultFormatter": "vscode.typescript-language-features"
  },
  "editor.unicodeHighlight.ambiguousCharacters": false,
  "pasteAndIndent.selectAfter": true,
  "gitlens.currentLine.enabled": false,
  "gitlens.hovers.currentLine.over": "line",
  "gitlens.codeLens.enabled": false,
  "python.formatting.blackArgs": ["--line-length", "110"],
  "blockman.n01LineHeight": 0,
  "workbench.colorCustomizations": {
    "activityBar.background": "#342C29",
    "titleBar.activeBackground": "#493D39",
    "titleBar.activeForeground": "#FBFAF9",
    "editor.lineHighlightBackground": "#1073cf2d",
    "editor.lineHighlightBorder": "#9fced11f"
  },
  "diffEditor.wordWrap": "off",
  "editor.guides.indentation": false,
  "editor.guides.bracketPairs": false,
  "blockman.n04ColorComboPreset": "Classic Light (Gradients)",
  "editor.inlayHints.enabled": "off",
  "editor.minimap.renderCharacters": false,
  "python.terminal.focusAfterLaunch": true,
  "editor.scrollbar.horizontal": "hidden",
  "cSpell.userWords": [
    "dtype",
    "gelu",
    "minigpt",
    "Multimodal",
    "ndarray",
    "onehot",
    "rcnn",
    "Shangyu",
    "softmax",
    "subsentence",
    "torchvision",
    "tqdm",
    "unimodal",
    "unsqueeze",
    "xingsy"
  ],
  "python.analysis.autoImportCompletions": true,
  "python.analysis.autoImportUserSymbols": true,
  "pythonIndent.trimLinesWithOnlyWhitespace": true,
  "pythonIndent.useTabOnHangingIndent": true,
  "editor.semanticTokenColorCustomizations": {
    "[Default Light+]": {
      "enabled": true,
      "rules": {
        "function.declaration": {
          "bold": true
        },
        "class.declaration": {
          "bold": true
        },
        "method.declaration": {
          "bold": true
        },
        "variable.declaration": {
          "italic": true
        },
        "property.declaration": {
          "italic": true
        }
      }
    }
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "isort.args": ["--profile", "black"],
  "editor.minimap.enabled": false,
  "python.analysis.typeCheckingMode": "basic",
  "git.openRepositoryInParentFolders": "never",
  "black-formatter.args": ["--line-length=110", "--skip-magic-trailing-comma"],
  "Codegeex.Privacy": false,
  "update.mode": "manual",
  "terminal.integrated.fontFamily": "consolas",
  "settingsSync.ignoredSettings": [
    "terminal.integrated.fontFamily",
    "terminal.integrated.fontSize",
    "editor.fontSize",
    "editor.fontFamily",
    "leetcode.defaultLanguage",
    "leetcode.workspaceFolder",
    "cmake.buildEnvironment"
  ],
  "settingsSync.ignoredExtensions": [
    "aminer.codegeex",
    "leetcode.vscode-leetcode",
    "ms-azuretools.vscode-docker",
    "ms-vscode-remote.remote-containers"
  ],
  "notebook.lineNumbers": "on",
  "window.dialogStyle": "custom",
  "window.titleBarStyle": "custom",
  "python.defaultInterpreterPath": "",
  "remote.SSH.serverInstallPath": {
    "titan-rtx": "/home/nfs04/xingsy/vscode",
    "2080ti-1": "/home/nfs04/xingsy/vscode",
    "2080ti-2": "/home/nfs04/xingsy/vscode",
    "3090ti-1": "/home/nfs04/xingsy/vscode",
    "3090ti-2": "/home/nfs04/xingsy/vscode",
    "v100": "/home/nfs04/xingsy/vscode",
    "v100-31": "/home/nfs04/xingsy/vscode"
  },
  "cmake.showOptionsMovedNotification": false,
  "Codegeex.Chat.LanguagePreference": "English",
  "files.eol": "\n",
  "files.trimFinalNewlines": true,
  "[html]": {
    "editor.defaultFormatter": "vscode.html-language-features"
  },
  "cmake.pinnedCommands": [
    "workbench.action.tasks.configureTaskRunner",
    "workbench.action.tasks.runTask"
  ],
  "explorer.confirmPasteNative": false,
  "Codegeex.License": "",
  "cursor.cpp.disabledLanguages": ["yaml", "cpp"],
  "blockman.n04Sub02ColorComboPresetForLightTheme": "Classic Light (Super gradients)",
  "blockman.n16EnableFocus": false,
  "qttools.extraSearchDirectories": ["D:/software/Qt/6.7.2/mingw_64"],
  "python.createEnvironment.trigger": "off"
}
```
