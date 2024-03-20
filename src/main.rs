use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
use tree_sitter::{Node, Parser};

static INDEX: AtomicUsize = AtomicUsize::new(0);
const CHARS: &[&str] = &["I", "J", "K", "L", "M"];

#[derive(Deserialize, Debug)]
struct HumanEval {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    test: String,
}

#[derive(Debug, Serialize)]
struct CobolEval {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    tests: Vec<CobolTest>,
}

#[derive(Debug)]
struct PyFunction {
    name: String,
    args: Vec<PyArgument>,
    return_type: PyType,
    docstring: String,
}

impl PyFunction {
    fn to_cobol(&self) -> Result<String> {
        let program_name = format_name(&self.name);
        let program_description = self.to_program_description();
        let working_storage_vars = self.to_linkage()?;

        Ok(prompt(
            &program_name,
            &program_description,
            &working_storage_vars,
        ))
    }

    fn to_linkage(&self) -> Result<String> {
        INDEX.store(0, Ordering::SeqCst);

        let args = self
            .args
            .iter()
            .map(PyArgument::to_cobol)
            .collect::<Result<Vec<_>>>()?;
        let args = args.join("\n");

        let return_type = self.return_type.to_cobol()?;
        let result_str = format!("           05 RESULT {}.", return_type);
        Ok(format!("       01 LINKED-ITEMS.\n{}\n{}", args, result_str))
    }

    fn to_program_description(&self) -> String {
        self.docstring
            .trim()
            .trim_matches('"')
            .lines()
            .map(str::trim)
            .map(|l| format!("      * {}", l))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug)]
struct PyArgument {
    name: String,
    type_: PyType,
}

impl PyArgument {
    fn to_cobol(&self) -> Result<String> {
        let type_ = self.type_.to_cobol()?;
        Ok(format!(
            "           05 L-{} {}.",
            format_name(&self.name),
            type_
        ))
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
enum PyType {
    Int,
    Float,
    String,
    Bool,
    Any,
    None,
    List(Box<PyType>),
}

impl PyType {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "int" => Ok(PyType::Int),
            "float" => Ok(PyType::Float),
            "str" | "Optional[str]" => Ok(PyType::String),
            "bool" => Ok(PyType::Bool),
            "Any" => Ok(PyType::Any),
            "List[int]" | "Tuple[int, int]" => Ok(PyType::List(Box::new(PyType::Int))),
            "List[float]" | "Tuple[float, float]" => Ok(PyType::List(Box::new(PyType::Float))),
            "List[str]" => Ok(PyType::List(Box::new(PyType::String))),
            "List[Any]" => Ok(PyType::List(Box::new(PyType::Any))),
            _ => Err(anyhow!("Could not parse type: {}", s)),
        }
    }

    fn from_node(node: &Node) -> Result<Self> {
        match node.kind() {
            "integer" => Ok(PyType::Int),
            "float" => Ok(PyType::Float),
            "string" => Ok(PyType::String),
            "none" => Ok(PyType::None),
            "true" | "false" => Ok(PyType::Bool),
            "list" | "tuple" => {
                let inner = node
                    .named_child(0)
                    .map(|n| PyType::from_node(&n))
                    .context("Could not get list inner type")??;
                Ok(PyType::List(Box::new(inner)))
            }
            "unary_operator" => Ok(node
                .named_child(0)
                .map(|n| PyType::from_node(&n))
                .context("Could not get unary inner type")??),
            _ => Err(anyhow!("Could not parse type: {}", node.kind())),
        }
    }

    fn to_cobol(&self) -> Result<String> {
        match self {
            PyType::Int => Ok("PIC S9(10)".into()),
            PyType::Float => Ok("COMP-2".into()),
            PyType::String => Ok("PIC X(100)".into()),
            PyType::Bool => Ok("PIC 9".into()),
            PyType::List(t) => {
                let inner = t.to_cobol()?;
                let suffix = CHARS[INDEX.fetch_add(1, Ordering::SeqCst)];
                Ok(format!("OCCURS 100 TIMES INDEXED BY N{} {}", suffix, inner))
            }
            PyType::Any => Err(anyhow!("Cannot convert Any to COBOL")),
            PyType::None => Err(anyhow!("Cannot convert None to COBOL")),
        }
    }
}

#[derive(Debug)]
struct PyTest {
    name: String,
    inputs: Vec<PyValue>,
    output: PyValue,
}

impl PyTest {
    fn to_cobol(&self, function: &PyFunction) -> Result<CobolTest> {
        let program_name = format_name(&self.name);
        let output_record = self.to_output_record(function)?;
        let working_storage = self.to_working_storage(function);
        let linked_items = function.to_linkage()?;
        let data_moves = self.to_data_moves(function);
        let write_logic = self.to_write_logic(function, &linked_items)?;

        Ok(CobolTest {
            test: eval_program(
                &program_name,
                &output_record,
                &working_storage,
                &linked_items,
                &data_moves,
                &write_logic,
            ),
            result: self.output.clone(),
        })
    }

    fn to_output_record(&self, function: &PyFunction) -> Result<String> {
        let return_type = match &function.return_type {
            PyType::List(t) => t.as_ref(),
            _ => &function.return_type,
        };

        let return_str = match return_type {
            PyType::Int => format!("{} SIGN LEADING", return_type.to_cobol()?),
            PyType::Float => "PIC X(15)".to_string(), // Float return hack
            _ => return_type.to_cobol()?,
        };

        Ok(format!("       01 OUTPUT-RECORD {}.", return_str))
    }

    fn to_working_storage(&self, function: &PyFunction) -> String {
        let return_float = match &function.return_type {
            PyType::Float => true,
            PyType::List(inner) if **inner == PyType::Float => true,
            _ => false,
        };

        if return_float {
            r#"       01 PARTS-REPR.
           05 DECIMAL-PART PIC X(9).
           05 FRACTIONAL-PART PIC X(5).
       01 FLOAT-REPR REDEFINES PARTS-REPR PIC S9(9)V9(5) SIGN LEADING."#
                .into()
        } else {
            "".into()
        }
    }

    fn to_write_logic(&self, function: &PyFunction, linkage: &str) -> Result<String> {
        match &function.return_type {
            PyType::List(_) => {
                let index = match_result_index(linkage)?;
                Ok(format!(
                    r#"       PERFORM VARYING N{index} FROM 1 BY 1 UNTIL N{index} > 100
           MOVE RESULT (N{index}) TO OUTPUT-RECORD
           WRITE OUTPUT-RECORD
       END-PERFORM"#
                ))
            }
            PyType::Float => Ok(r#"       MOVE RESULT TO FLOAT-REPR
       STRING DECIMAL-PART "." FRACTIONAL-PART INTO OUTPUT-RECORD
       WRITE OUTPUT-RECORD"#
                .to_string()),
            _ => {
                Ok("       MOVE RESULT TO OUTPUT-RECORD\n       WRITE OUTPUT-RECORD\n".to_string())
            }
        }
    }

    fn to_data_moves(&self, function: &PyFunction) -> String {
        function
            .args
            .iter()
            .zip(self.inputs.iter())
            .flat_map(|(arg, inp)| {
                let arg_name = format_name(&arg.name);
                match &arg.type_ {
                    PyType::List(_) => inp
                        .parse_list()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            format!(
                                "       MOVE {v} TO L-{arg_name}({i})",
                                i = i + 1,
                                arg_name = arg_name
                            )
                        })
                        .collect::<Vec<_>>(),
                    _ => vec![format!(
                        "       MOVE {inp} TO L-{arg_name}",
                        inp = inp.value
                    )],
                }
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
}

#[derive(Debug, Clone, Serialize)]
struct PyValue {
    value: String,
    type_: PyType,
}

impl PyValue {
    fn from_node(s: &Node, src: &str) -> Result<PyValue> {
        let value = s.utf8_text(src.as_bytes()).context("Could not get value")?;
        Ok(PyValue {
            value: value.to_string(),
            type_: PyType::from_node(s)?,
        })
    }

    fn from_node_and_type(s: &Node, src: &str, type_: &PyType) -> Result<PyValue> {
        let value = s.utf8_text(src.as_bytes()).context("Could not get value")?;
        Ok(PyValue {
            value: value.to_string(),
            type_: type_.clone(),
        })
    }

    fn parse_list(&self) -> Result<Vec<String>> {
        match &self.type_ {
            PyType::List(_) => Ok(self
                .value
                .trim_matches(|c| c == '[' || c == ']' || c == '(' || c == ')')
                .split(',')
                .map(|s| s.trim_start())
                .map(String::from)
                .collect::<Vec<_>>()),
            _ => Err(anyhow!("Cannot parse non-list type")),
        }
    }
}

#[derive(Debug, Serialize)]
struct CobolTest {
    test: String,
    result: PyValue,
}

fn format_name(name: &str) -> String {
    name.to_uppercase().replace('_', "-")
}

fn match_result_index(linkage: &str) -> Result<String> {
    let regex_pattern = r"           05 RESULT OCCURS.*TIMES INDEXED BY N(\w)";
    let re = regex::Regex::new(regex_pattern)?;
    re.captures(linkage)
        .and_then(|caps: regex::Captures<'_>| caps.get(1).map(|m| m.as_str().to_string()))
        .context("Could not find index")
}

fn load_human_eval(path: &Path) -> Vec<HumanEval> {
    let file = File::open(path).expect("Could not open file");
    let reader = BufReader::new(file);

    reader
        .lines()
        .map(|l| l.expect("Could not parse line"))
        .map(|l| serde_json::from_str::<HumanEval>(&l).expect("Could not parse JSON"))
        .collect()
}

fn get_child_by_field_name<'a>(
    field_name: &str,
    node: &'a Node<'a>,
    src: &'a str,
) -> Result<&'a str> {
    node.child_by_field_name(field_name)
        .and_then(|n| n.utf8_text(src.as_bytes()).ok())
        .ok_or_else(|| anyhow!("Field {} not found", field_name))
}

fn parse(parser: &mut Parser, src: &str) -> Result<Vec<PyFunction>> {
    let tree = parser.parse(src, None).context("Could not parse")?;
    let root_node = tree.root_node();

    let query = tree_sitter::Query::new(
        tree_sitter_python::language(),
        "(function_definition) @func",
    )?;
    let mut query_cursor = tree_sitter::QueryCursor::new();
    let matches = query_cursor.captures(&query, root_node, src.as_bytes());

    matches
        .flat_map(|(match_, _)| {
            match_.captures.iter().map(move |capture| {
                let func_node = capture.node;
                let func_name = get_child_by_field_name("name", &func_node, src)?;
                let return_type = get_child_by_field_name("return_type", &func_node, src)?;
                let docstring = get_child_by_field_name("body", &func_node, src)?;

                let typed_params = func_node
                    .child_by_field_name("parameters")
                    .context("Parameters not found")?
                    .named_children(&mut func_node.walk())
                    .filter(|n| n.kind() == "typed_parameter")
                    .map(|n| {
                        let name = n
                            .child(0)
                            .and_then(|n| n.utf8_text(src.as_bytes()).ok())
                            .context("Could not get name")?;
                        let type_ = get_child_by_field_name("type", &n, src)?;
                        Ok(PyArgument {
                            name: name.to_string(),
                            type_: PyType::from_str(type_)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(PyFunction {
                    name: func_name.to_string(),
                    args: typed_params,
                    return_type: PyType::from_str(return_type).unwrap(),
                    docstring: docstring.to_string(),
                })
            })
        })
        .collect()
}

fn parse_test(
    parser: &mut Parser,
    src: &str,
    name: &str,
    return_type: &PyType,
) -> Result<Vec<Result<PyTest>>> {
    let tree = parser.parse(src, None).context("Could not parse")?;
    let root_node = tree.root_node();

    let query = tree_sitter::Query::new(
        tree_sitter_python::language(),
        "(assert_statement
 (comparison_operator) @assert)
",
    )?;
    let mut query_cursor = tree_sitter::QueryCursor::new();
    let matches = query_cursor.captures(&query, root_node, src.as_bytes());

    let tests = matches
        .flat_map(|(match_, _)| {
            match_.captures.iter().map(move |capture| {
                let assert_node = capture.node;

                // First child of comparison operator is the argument list
                let argument_list_result = assert_node
                    .named_child(0)
                    .and_then(|n| n.child_by_field_name("arguments"))
                    .ok_or_else(|| anyhow::anyhow!("No arguments found"))
                    .and_then(|args| {
                        args.named_children(&mut assert_node.walk())
                            .map(|n| PyValue::from_node(&n, src))
                            .collect::<Result<Vec<PyValue>>>()
                            .context("Could not parse arguments")
                    });

                // Second child is the result
                let result = assert_node
                    .named_child(1)
                    .map(|n| PyValue::from_node_and_type(&n, src, return_type))
                    .ok_or_else(|| anyhow::anyhow!("Could not get result"))
                    .and_then(|res| res.context("Could not parse result"));

                match (argument_list_result, result) {
                    (Ok(argument_list), Ok(result)) => Ok(PyTest {
                        name: name.to_string(),
                        inputs: argument_list,
                        output: result,
                    }),
                    (Err(e), _) | (_, Err(e)) => Err(e),
                }
            })
        })
        .collect();

    Ok(tests)
}

pub fn prompt(program_name: &str, program_description: &str, linkage_vars: &str) -> String {
    format!(
        r#"       IDENTIFICATION DIVISION.
       PROGRAM-ID. {program_name}.

       ENVIRONMENT DIVISION.
       
       INPUT-OUTPUT SECTION.

       DATA DIVISION.

       LINKAGE SECTION.

{linkage_vars}

{program_description}

      * Complete the WORKING-STORAGE SECTION and the PROCEDURE DIVISION
      * Store the result in the RESULT variable and mark the end of your program with END PROGRAM

       WORKING-STORAGE SECTION.
       
       "#
    )
}

pub fn eval_program(
    program_name: &str,
    output_record: &str,
    working_storage: &str,
    linked_items: &str,
    data_moves: &str,
    write_logic: &str,
) -> String {
    format!(
        r#"       IDENTIFICATION DIVISION.
       PROGRAM-ID. {program_name}-CALL.

       ENVIRONMENT DIVISION.

       INPUT-OUTPUT SECTION.

       FILE-CONTROL.

       SELECT OUTPUT-FILE ASSIGN TO "{program_name}.TXT"
           ORGANIZATION IS LINE SEQUENTIAL
           STATUS IS OUTPUT-FILE-STATUS.
       
       DATA DIVISION.

       FILE SECTION.
       FD OUTPUT-FILE.
{output_record}

       WORKING-STORAGE SECTION.

       01 OUTPUT-FILE-STATUS PIC X(02).

{working_storage}

{linked_items}

       PROCEDURE DIVISION.

{data_moves}

       CALL "{program_name}" USING LINKED-ITEMS
       
       OPEN OUTPUT OUTPUT-FILE

       IF OUTPUT-FILE-STATUS NOT = "00"
           DISPLAY "ERROR OPENING OUTPUT FILE"
           STOP RUN
        END-IF

{write_logic}

        IF OUTPUT-FILE-STATUS NOT = "00"
            DISPLAY "ERROR WRITING TO OUTPUT FILE"
            STOP RUN
        END-IF

        CLOSE OUTPUT-FILE
        .
       "#
    )
}

fn main() -> Result<()> {
    let human_eval_path = Path::new("./data/HumanEval.jsonl");
    let human_eval = load_human_eval(human_eval_path);
    let mut cobol_evals = vec![];

    let mut parser = Parser::new();
    parser.set_language(tree_sitter_python::language())?;

    for eval in human_eval {
        let functions = parse(&mut parser, &eval.prompt);
        match functions {
            Ok(f) => {
                if f.len() > 1 {
                    continue;
                }
                let function = f.into_iter().next().unwrap();

                match function.to_cobol() {
                    Ok(prompt) => {
                        let tests = parse_test(
                            &mut parser,
                            &eval.test,
                            &eval.entry_point,
                            &function.return_type,
                        );

                        let valid_tests = match tests {
                            Ok(ts) => ts
                                .into_iter()
                                .filter_map(|t| {
                                    let converted = t.and_then(|t| t.to_cobol(&function));
                                    if let Err(e) = converted {
                                        println!(
                                            "Could not convert to COBOL: {}",
                                            eval.entry_point
                                        );
                                        println!("Error: {}\n", e);
                                        None
                                    } else {
                                        converted.ok()
                                    }
                                })
                                .collect::<Vec<_>>(),
                            Err(e) => {
                                println!("{}", eval.test);
                                println!("Could not parse tests: {}", eval.entry_point);
                                println!("Error: {}\n", e);
                                vec![]
                            }
                        };

                        if valid_tests.is_empty() {
                            println!("No valid tests for: {}\n", eval.entry_point);
                            continue;
                        }

                        cobol_evals.push(CobolEval {
                            task_id: eval.task_id,
                            prompt,
                            entry_point: eval.entry_point,
                            canonical_solution: eval.canonical_solution,
                            tests: valid_tests,
                        });
                    }
                    Err(e) => {
                        println!("Could not convert to COBOL: {}", eval.entry_point);
                        println!("Error: {}\n", e);
                        continue;
                    }
                }
            }
            Err(e) => {
                println!("Could not parse: {}", eval.entry_point);
                println!("Error: {}\n", e);
                continue;
            }
        }
    }

    let cobol_evals_path = Path::new("./data/CobolEval.jsonl");
    let mut cobol_evals_file = File::create(cobol_evals_path).expect("Could not create file");
    for eval in cobol_evals {
        let json = serde_json::to_string(&eval).expect("Could not serialize");
        writeln!(cobol_evals_file, "{}", json).expect("Could not write to file");
    }

    Ok(())
}
