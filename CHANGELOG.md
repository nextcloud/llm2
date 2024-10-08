<!--
  - SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


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