# Security Policy

## Supported versions

Until the project has multiple maintained releases, security fixes target the latest code on `main` and the latest published release when practical.

## Reporting a vulnerability

Please **do not open a public issue** for a vulnerability that could put users at risk before a fix is available.

Use GitHub's private vulnerability reporting feature for this repository if it is enabled. If private reporting is not available, contact the repository maintainer privately through the contact method listed on the maintainer's GitHub profile and include `bounded-moe security` in the subject/message.

Please include:

- affected version or commit;
- impact and threat model;
- reproduction steps or proof of concept;
- relevant configuration and platform details;
- any suggested mitigation, if known.

Do not include real credentials, API keys, private model data, or other people's personal information in a report.

## Scope notes

bounded-moe modifies low-level model-loading, memory, file-I/O, and compute paths. Crashes and correctness bugs are welcome as normal bug reports unless they create a security boundary bypass, arbitrary code execution, unintended file access, secret exposure, or another security impact.
