def safe_relative:
  type == "string"
  and length > 0
  and (startswith("/") | not)
  and (split("/") | all(. != ".."));

.schema_version == 1
and .platform == "linux-x86_64"
and (.tools | type == "array" and length > 0)
and (([.tools[].id] | unique | length) == (.tools | length))
and all(.tools[];
  (.id | type == "string" and test("^[a-z0-9][a-z0-9-]*$"))
  and (.version | type == "string" and length > 0)
  and (.url | type == "string" and startswith("https://"))
  and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.archive == "tar.gz" or .archive == "tar.xz" or .archive == "zip")
  and (.strip_components | type == "number" and . >= 0 and floor == .)
  and (.probe | safe_relative)
  and (.path | safe_relative)
  and ((.environment // {}) | type == "object")
  and all((.environment // {}) | to_entries[];
    (.key | test("^[A-Z][A-Z0-9_]*$"))
    and (.value | safe_relative))
  and (.license | type == "string" and length > 0)
  and ((.runtime_license // "UNSET") | test("^(UNSET|[A-Z][A-Z0-9_]*)$"))
)
# Wheels are pip-installed rather than unpacked onto PATH, so they carry the
# provenance fields only (see AmbiqAI/ns-cmsis-nn#394).
and (.python_tools | type == "array" and length > 0)
and (([.python_tools[].id] | unique | length) == (.python_tools | length))
and all(.python_tools[];
  (.id | type == "string" and test("^[a-z0-9][a-z0-9-]*$"))
  and (.version | type == "string" and length > 0)
  and (.url | type == "string" and startswith("https://files.pythonhosted.org/"))
  # A pythonhosted URL is content-addressed, so the recorded version is
  # otherwise unchecked against the file it names; the wheel filename carries
  # the distribution and version (PEP 427), with dashes escaped as underscores.
  and (. as $tool
    | $tool.url
    | split("/")
    | last
    | startswith(($tool.id | gsub("-"; "_")) + "-" + $tool.version + "-"))
  and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.license | type == "string" and length > 0)
)
