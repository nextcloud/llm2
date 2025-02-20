<!--
  - SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.3.1] -2025-02-20

### Fixed

* fix(ChatProcessor): Fix attribute access


## [2.3.0] -2025-02-18

### New

* enh: Always use the chat method
* feat: add changetone provider
* feat: add proofread provider

### Fixed

* fix: Rework task_processor loading to wait until model is loaded


## [2.2.2] - 2024-12-20

### Fixed

* fix(chatwithtools): Expect a list of tool messages
* fix(main): Don't drop background task when app is disabled
* fix: Only run background thread once
* fix(summarize): Improve prompt
* fix(chat): Expect json-stringified messages in history {role, content}

## [2.2.1] - 2024-12-16

### Fixed

- fixed failed import


## [2.2.0] - 2024-12-12

### New

enh(summarize): Try to make it use bulleted lists for longer texts
enh: Implement chatwithtools task type
feat: Use chunking if the text doesn't fit the context

### Fixed

fix(summarize): Use a better algorithm for chunked summaries
fix(summarize): Always summarize at least once
fix(ci): app_api is pre-installed from NC 31 (#37) Anupam Kumar* 03.10.24, 14:13

## [2.1.4] - 2024-09-12

### Fix

- update docker image version


## [2.1.3] - 2024-09-11

### Fix

- update context size for llama 3.1


## [2.1.2] - 2024-08-26

### Fix

- filename of the llama3.1 model in config
- catch JSONDecodeError for when server is in maintenance mode

### Change

- better app_enabled handling


## [2.1.1] - 2024-08-23

### Fix

- compare uppercase COMPUTE_DEVICE value (#27)


## [2.1.0] - 2024-08-22

### Fix

- Catch network exceptions and keep the loop going

### Change

- Migrate default config to llama compatible config
- Use COMPUTE_DEVICE to determine gpu offloading
- Use TaskProcessingProvider class for registration
- Better handling of app enabled state
- Download models on /init


## [2.0.1] - 2024-08-14

### Fix

- Disable ContextWrite chain as it does not work with Llama 3 / 3.1

## [2.0.0] - 2024-08-13

## Breaking Change

- Requires Nextcloud 30 and AppAPI v3

### New

feat: Update prompts and add new task types
feat: Add task processing API