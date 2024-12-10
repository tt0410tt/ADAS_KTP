def calculate(num1, num2, operation):
    """Perform basic arithmetic operations."""
    try:
        num1 = float(num1)
        num2 = float(num2)

        if operation == 'add':
            return num1 + num2
        elif operation == 'subtract':
            return num1 - num2
        elif operation == 'multiply':
            return num1 * num2
        elif operation == 'divide':
            if num2 != 0:
                return num1 / num2
            return "Error: Division by zero"
        else:
            return "Error: Invalid operation"
    except ValueError:
        return "Error: Invalid input"
