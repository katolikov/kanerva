

def test_cpp_facts_read_templates_classes_and_overloads():
    from experiments.repomem import extract_cpp_facts

    code = """
class Reader {
 public:
  template <typename T> T get(int a);
  void over(int a);
  void over(double a, int b);
};
template <class T>
inline T twice(const T& a) { return a; }
"""
    facts = extract_cpp_facts("include/reader.h", code)
    assert facts["sig:Reader::get"] == ("signature", "Reader::get", "int a")
    assert facts["sig:Reader::over"] == ("signature", "Reader::over", "double a, int b | int a")
    assert facts["sig:twice"] == ("signature", "twice", "const T& a")
