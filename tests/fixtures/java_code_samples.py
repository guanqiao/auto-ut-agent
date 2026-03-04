"""Java code samples for testing.

This module provides various Java code samples for use in tests.
"""

# Simple Java class
SIMPLE_JAVA_CLASS = """
package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public double divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Cannot divide by zero");
        }
        return (double) a / b;
    }
}
"""

# Java class with dependencies
JAVA_CLASS_WITH_DEPENDENCIES = """
package com.example;

import java.util.List;
import java.util.ArrayList;
import java.util.Optional;

public class DataProcessor {
    private List<String> data;
    private final Validator validator;
    
    public DataProcessor(Validator validator) {
        this.data = new ArrayList<>();
        this.validator = validator;
    }
    
    public void addData(String item) {
        if (validator.isValid(item)) {
            data.add(item);
        }
    }
    
    public Optional<String> getData(int index) {
        if (index >= 0 && index < data.size()) {
            return Optional.of(data.get(index));
        }
        return Optional.empty();
    }
    
    public int getCount() {
        return data.size();
    }
    
    public interface Validator {
        boolean isValid(String item);
    }
}
"""

# Java interface
JAVA_INTERFACE = """
package com.example;

public interface CalculatorService {
    int add(int a, int b);
    int subtract(int a, int b);
    int multiply(int a, int b);
    double divide(int a, int b);
    double power(double base, double exponent);
}
"""

# Java abstract class
JAVA_ABSTRACT_CLASS = """
package com.example;

public abstract class AbstractCalculator {
    protected String name;
    
    public AbstractCalculator(String name) {
        this.name = name;
    }
    
    public abstract int calculate(int a, int b);
    
    public String getName() {
        return name;
    }
    
    protected void logOperation(String operation) {
        System.out.println("Operation: " + operation);
    }
}
"""

# Java enum
JAVA_ENUM = """
package com.example;

public enum Operation {
    ADD("+", (a, b) -> a + b),
    SUBTRACT("-", (a, b) -> a - b),
    MULTIPLY("*", (a, b) -> a * b),
    DIVIDE("/", (a, b) -> b != 0 ? a / b : 0);
    
    private final String symbol;
    private final OperationFunction function;
    
    Operation(String symbol, OperationFunction function) {
        this.symbol = symbol;
        this.function = function;
    }
    
    public int apply(int a, int b) {
        return function.apply(a, b);
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    @FunctionalInterface
    interface OperationFunction {
        int apply(int a, int b);
    }
}
"""

# Java exception
JAVA_EXCEPTION = """
package com.example;

public class CalculationException extends RuntimeException {
    private final String operation;
    private final int operand1;
    private final int operand2;
    
    public CalculationException(String message, String operation, int operand1, int operand2) {
        super(message);
        this.operation = operation;
        this.operand1 = operand1;
        this.operand2 = operand2;
    }
    
    public String getOperation() {
        return operation;
    }
    
    public int getOperand1() {
        return operand1;
    }
    
    public int getOperand2() {
        return operand2;
    }
}
"""

# Java class with generics
JAVA_GENERICS_CLASS = """
package com.example;

import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;

public class GenericContainer<T> {
    private List<T> items;
    
    public GenericContainer() {
        this.items = new ArrayList<>();
    }
    
    public void add(T item) {
        items.add(item);
    }
    
    public List<T> filter(Predicate<T> predicate) {
        List<T> result = new ArrayList<>();
        for (T item : items) {
            if (predicate.test(item)) {
                result.add(item);
            }
        }
        return result;
    }
    
    public int size() {
        return items.size();
    }
}
"""

# Java class with static methods
JAVA_STATIC_CLASS = """
package com.example;

public final class MathUtils {
    
    private MathUtils() {
        // Utility class
    }
    
    public static int factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative");
        }
        if (n == 0 || n == 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
    
    public static boolean isPrime(int n) {
        if (n <= 1) {
            return false;
        }
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }
    
    public static int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }
}
"""

# Java class with annotations
JAVA_ANNOTATED_CLASS = """
package com.example;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class UserService {
    
    private final UserRepository userRepository;
    
    @Autowired
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    public User findById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new UserNotFoundException("User not found: " + id));
    }
    
    public User createUser(String name, String email) {
        User user = new User();
        user.setName(name);
        user.setEmail(email);
        return userRepository.save(user);
    }
}
"""


def get_java_class_with_method(method_name: str) -> str:
    """Get a Java class with a specific method for testing.
    
    Args:
        method_name: Name of the method to include
        
    Returns:
        Java class code with the specified method
    """
    methods = {
        "add": """
    public int add(int a, int b) {
        return a + b;
    }
""",
        "subtract": """
    public int subtract(int a, int b) {
        return a - b;
    }
""",
        "divide": """
    public double divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Cannot divide by zero");
        }
        return (double) a / b;
    }
""",
    }
    
    method_code = methods.get(method_name, "")
    
    return f"""
package com.example;

public class Calculator {{
{method_code}
}}
"""
