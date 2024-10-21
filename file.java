////////////////////////////SLIP NO.1/////////////////
// Q1. Write a program to accept a number from user and generate multiplication table of a
// number.
// [10 Marks]

import java.util.Scanner;
public class MultiplicationTable {
public static void main(String[] args) {
// Create a Scanner object to read input
Scanner scanner = new Scanner(System.in);
// Prompt the user for a number
System.out.print("Enter a number to generate its multiplication table: ");
int number = scanner.nextInt();
// Generate and print the multiplication table
System.out.println("Multiplication Table of " + number + ":");
for (int i = 1; i <= 10; i++) {
System.out.println(number + " x " + i + " = " + (number * i));
}
// Close the scanner
scanner.close();
}
}

/*OUTPUT
buntu@ubuntu-OptiPlex-5000:~$ javac MultiplicationTable.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MultiplicationTable
Enter a number to generate its multiplication table: 5
Multiplication Table of 5:
5x1=5
5 x 2 = 10
5 x 3 = 15
5 x 4 = 20
5 x 5 = 25
5 x 6 = 30
5 x 7 = 35
5 x 8 = 40
5 x 9 = 45
5 x 10 = 50
*/

// Q2. Construct a linked List containing names of colours: red, blue, yellow and orange. Then
// extend the program to do the following:
// i.Display the contents of the List using an Iterator
// ii.Display the contents of the List in reverse order using a ListIterator
// iii.Create another list containing pink and green. Insert the elements of this list between
// blue and yellow.
// [20 Marks]

import java.util.LinkedList;
import java.util.Iterator;
import java.util.ListIterator;public class ColorLinkedList {
public static void main(String[] args) {
// Step 1: Create a linked list containing initial colors
LinkedList<String> colors = new LinkedList<>();
colors.add("red");
colors.add("blue");
colors.add("yellow");
colors.add("orange");
// i. Display the contents of the list using an Iterator
System.out.println("Colors in the list:");
Iterator<String> iterator = colors.iterator();
while (iterator.hasNext()) {
System.out.println(iterator.next());
}
// ii. Display the contents of the list in reverse order using a ListIterator
System.out.println("\nColors in reverse order:");
ListIterator<String> listIterator = colors.listIterator(colors.size());
while (listIterator.hasPrevious()) {
System.out.println(listIterator.previous());
}
// iii. Create another list containing pink and green
LinkedList<String> newColors = new LinkedList<>();
newColors.add("pink");
newColors.add("green");
// Insert the elements of this list between blue and yellow
int indexToInsert = colors.indexOf("yellow");
colors.addAll(indexToInsert, newColors);
// Display the updated list
System.out.println("\nUpdated list of colors:");
for (String color : colors) {
System.out.println(color);
}
}
}

/*OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac ColorLinkedList.java
ubuntu@ubuntu-OptiPlex-5000:~$ java ColorLinkedList
Colors in the list:
red
blue
yellow
orange
Colors in reverse order:
orange
yellow
blue
red
Updated list of colors:red
blue
pink
green
yellow
orange
*/
//////////////////////////////////////////////SLIP NO 2//////////////////////
// Q1. Write a program to accept ‘n’ integers from the user & store them in an Array List collection.
// Display the elements of Array List.
// [10 Marks]

import java.util.ArrayList;
import java.util.Scanner;
public class ArrayListExample {
public static void main(String[] args) {
// Create a Scanner object to read input
Scanner scanner = new Scanner(System.in);
// Create an ArrayList to store integers
ArrayList<Integer> numbers = new ArrayList<>();
// Prompt the user for the number of integers to accept
System.out.print("Enter the number of integers you want to input: ");
int n = scanner.nextInt();
// Accept ’n’ integers from the user
System.out.println("Please enter " + n + " integers:");
for (int i = 0; i < n; i++) {
int num = scanner.nextInt();
numbers.add(num); // Add the number to the ArrayList
}
// Display the elements of the ArrayList
System.out.println("Elements in the ArrayList:");
for (Integer number : numbers) {
System.out.println(number);
}
// Close the scanner
scanner.close();
}
}

/*OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac ArrayListExample.java
ubuntu@ubuntu-OptiPlex-5000:~$ java ArrayListExample
Enter the number of integers you want to input: 5
Please enter 5 integers:
10
20
30
4050
Elements in the ArrayList:
10
20
30
40
50
ubuntu@ubuntu-OptiPlex-5000:~$ javac ArrayListExample.java
ubuntu@ubuntu-OptiPlex-5000:~$ java ArrayListExample
Enter the number of integers you want to input: 5
Please enter 5 integers:
50
20
30
40
10
Elements in the ArrayList:
50
20
30
40
10
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Define a class MyNumber having one private integer data member. Write a default constructor
// initialize it to 0 and another constructor to initialize it to a value. Write methods isNegative,
// isPositive, isOdd, iseven. Use command line argument to pass a value to the object and perform
// the above operations.
// [20 Marks]

public class MyNumber {
private int value;
// Default constructor
public MyNumber() {
this.value = 0;
}
// Constructor to initialize the value
public MyNumber(int value) {
this.value = value;
}
// Method to check if the number is negative
public boolean isNegative() {
return value < 0;
}
// Method to check if the number is positive
public boolean isPositive() {
return value > 0;
}// Method to check if the number is odd
public boolean isOdd() {
return value % 2 != 0;
}
// Method to check if the number is even
public boolean isEven() {
return value % 2 == 0;
}
public static void main(String[] args) {
// Check if a command line argument is provided
if (args.length < 1) {
System.out.println("Please provide a number as a command line argument.");
return;
}
try {
// Parse the command line argument to an integer
int number = Integer.parseInt(args[0]);
// Create an object of MyNumber
MyNumber myNumber = new MyNumber(number);
// Perform operations
System.out.println("Value: " + number);
System.out.println("Is Negative: " + myNumber.isNegative());
System.out.println("Is Positive: " + myNumber.isPositive());
System.out.println("Is Odd: " + myNumber.isOdd());
System.out.println("Is Even: " + myNumber.isEven());
} catch (NumberFormatException e) {
System.out.println("Please provide a valid integer.");
}
}
}

/*OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac MyNumber.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MyNumber 0
Value: 0
Is Negative: false
Is Positive: false
Is Odd: false
Is Even: true
ubuntu@ubuntu-OptiPlex-5000:~$
*/

/////////////////////////////slip3////////////////////////////////

// Q1

import java.util.Scanner;

public class Slip3 {
    public static void main(String[] args) {
        // Q1. Write a program to accept the 'n' different numbers from user and store
        // it in array. Also print
        // the sum of elements of the array.
        Scanner s = new Scanner(System.in);

        System.out.print("enter total number : ");
        int totalnum = s.nextInt();

        int[] arr = new int[totalnum];

        for (int i = 0; i < totalnum; i++) {
            System.out.print("enter no."+(i+1)+" : ");
            int num = s.nextInt();
            arr[i] = num;
        }

        int sum = 0;
        for (int i = 0; i < totalnum; i++) {
            sum += arr[i];
        }

        System.out.println("sum of all number in array -> " + sum);

        s.close();
    }
}

// Q2. Write a program to copy the contents from one file into another file in upper case.

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class slip3Q2 {

    public static void main(String[] args) {
    String inputfilepath = "/home/pranav/PROGRAMMING/practical_slips/javaSlips/inputfile.txt";
    String outputfilepath = "/home/pranav/PROGRAMMING/practical_slips/javaSlips/outputfile.txt"; 
        
        try(FileReader reader = new FileReader(inputfilepath);
            FileWriter writer = new FileWriter(outputfilepath)){

                int character;

                while ((character = reader.read()) != -1) {
                    writer.write(Character.toUpperCase(character));
                }
                System.out.println("FILE COPIED SUCCESSFULLY");
            } catch(IOException e){
                System.out.println("an error occurred "+e.getMessage());
            }
    }
}

//////////////////////////////SLIP NO 4///////////////////////////
// Q1. Write a program to accept the user name and greets the user by name. Before displaying the
// user’s name, convert it to upper case letters. For example, if the user’s name is Raj, then display
// greet message as "Hello, RAJ, nice to meet you!".
// [10 Marks]

import java.util.Scanner;public class GreetingUser {
public static void main(String[] args) {
// Create a Scanner object to read input
Scanner scanner = new Scanner(System.in);
// Prompt the user for their name
System.out.print("Enter your name: ");
String name = scanner.nextLine();
// Convert the name to uppercase
String upperCaseName = name.toUpperCase();
// Display the greeting message
System.out.println("Hello, " + upperCaseName + ", nice to meet you!");
// Close the scanner
scanner.close();
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac GreetingUser.java
ubuntu@ubuntu-OptiPlex-5000:~$ java GreetingUser
Enter your name: raj
Hello, RAJ, nice to meet you!
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Write a program which define class Product with data member as id, name and price. Store
// the information of 5 products and Display the name of product having minimum price (Use array
// of object).
// [20 Marks]
import java.util.Scanner;
class Product {
private int id;
private String name;
private double price;
// Constructor
public Product(int id, String name, double price) {
this.id = id;
this.name = name;
this.price = price;
}
// Getter for price
public double getPrice() {
return price;
}
// Getter for name
public String getName() {return name;
}
}
public class ProductMinPrice {
public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
Product[] products = new Product[5];
// Input information for 5 products
for (int i = 0; i < 5; i++) {
System.out.print("Enter id for product " + (i + 1) + ": ");
int id = scanner.nextInt();
scanner.nextLine(); // Consume newline
System.out.print("Enter name for product " + (i + 1) + ": ");
String name = scanner.nextLine();
System.out.print("Enter price for product " + (i + 1) + ": ");
double price = scanner.nextDouble();
products[i] = new Product(id, name, price);
}
// Find the product with the minimum price
Product minPriceProduct = products[0];
for (int i = 1; i < products.length; i++) {
if (products[i].getPrice() < minPriceProduct.getPrice()) {
minPriceProduct = products[i];
}
}
// Display the name of the product with the minimum price
System.out.println("Product with minimum price: " + minPriceProduct.getName());
// Close the scanner
scanner.close();
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac ProductMinPrice.java
ubuntu@ubuntu-OptiPlex-5000:~$ java ProductMinPrice
Enter id for product 1: 110
Enter name for product 1: Dell
Enter price for product 1: 40000
Enter id for product 2: 120
Enter name for product 2: Lenovo
Enter price for product 2: 60000
Enter id for product 3: 130
Enter name for product 3: Apple
Enter price for product 3: 80000
Enter id for product 4: 140
Enter name for product 4: Intel
Enter price for product 4: 40000
Enter id for product 5: 150Enter name for product 5: Wipro
Enter price for product 5: 55000
Product with minimum price: Dell
ubuntu@ubuntu-OptiPlex-5000:~$
*/
//////////////////////////////SLIP NO 6///////////////////////////
// Q1. Accept ’n’ integers from the user and store them in a collection. Display them in the sorted order.
// The collection should not accept duplicate elements. (Use a suitable collection). Search for a
// particular element using predefined search method in the Collection framework. [10 Marks]

import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;
public class UniqueSortedIntegers {
public static void main(String[] args) {
// Create a Scanner object to read input
Scanner scanner = new Scanner(System.in);
// Create a TreeSet to store unique integers
Set<Integer> numbers = new TreeSet<>();
// Prompt the user for the number of integers to accept
System.out.print("Enter the number of integers you want to input: ");
int n = scanner.nextInt();
// Accept ’n’ integers from the user
System.out.println("Please enter " + n + " integers (duplicates will be ignored):");
for (int i = 0; i < n; i++) {
int num = scanner.nextInt();
numbers.add(num); // Add the number to the TreeSet
}
// Display the sorted elements of the TreeSet
System.out.println("Sorted unique integers:");
for (Integer number : numbers) {
System.out.println(number);
}
// Search for a particular element
System.out.print("Enter an integer to search for: ");
int searchNum = scanner.nextInt();
// Check if the element is present in the collection
if (numbers.contains(searchNum)) {
System.out.println(searchNum + " is present in the collection.");
} else {
System.out.println(searchNum + " is not present in the collection.");
}
// Close the scanner
scanner.close();
}
}

/* OUTPUT
buntu@ubuntu-OptiPlex-5000:~$ javac UniqueSortedIntegers.javaubuntu@ubuntu-OptiPlex-5000:~$ java UniqueSortedIntegers
Enter the number of integers you want to input: 5
Please enter 5 integers (duplicates will be ignored):
50
30
20
40
10
Sorted unique integers:
10
20
30
40
50
Enter an integer to search for: 30
30 is present in the collection.
ubuntu@ubuntu-OptiPlex-5000:~$ javac UniqueSortedIntegers.java
ubuntu@ubuntu-OptiPlex-5000:~$ java UniqueSortedIntegers
Enter the number of integers you want to input: 5
Please enter 5 integers (duplicates will be ignored):
50
30
40
10
20
Sorted unique integers:
10
20
30
40
50
Enter an integer to search for: 60
60 is not present in the collection.
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Write a program which define class Employee with data member as id, name and salary Store
// the information of ’n’ employees and Display the name of employee having maximum salary (Use
// array of object).
// [20 Marks]
import java.util.Scanner;
class Employee {
private int id;
private String name;
private double salary;
// Constructor
public Employee(int id, String name, double salary) {
this.id = id;
this.name = name;
this.salary = salary;
}// Getter for salary
public double getSalary() {
return salary;
}
// Getter for name
public String getName() {
return name;
}
}
public class EmployeeMaxSalary {
public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
// Prompt the user for the number of employees
System.out.print("Enter the number of employees: ");
int n = scanner.nextInt();
// Create an array of Employee objects
Employee[] employees = new Employee[n];
// Input information for ’n’ employees
for (int i = 0; i < n; i++) {
System.out.print("Enter ID for employee " + (i + 1) + ": ");
int id = scanner.nextInt();
scanner.nextLine(); // Consume newline
System.out.print("Enter name for employee " + (i + 1) + ": ");
String name = scanner.nextLine();
System.out.print("Enter salary for employee " + (i + 1) + ": ");
double salary = scanner.nextDouble();
employees[i] = new Employee(id, name, salary);
}
// Find the employee with the maximum salary
Employee maxSalaryEmployee = employees[0];
for (int i = 1; i < employees.length; i++) {
if (employees[i].getSalary() > maxSalaryEmployee.getSalary()) {
maxSalaryEmployee = employees[i];
}
}
// Display the name of the employee with the maximum salary
System.out.println("Employee with the maximum salary: " + maxSalaryEmployee.getName());
// Close the scanner
scanner.close();
}
}

/*OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac EmployeeMaxSalary.java
ubuntu@ubuntu-OptiPlex-5000:~$ java EmployeeMaxSalary
Enter the number of employees: 5Enter ID for employee 1: 111
Enter name for employee 1: Ram
Enter salary for employee 1: 50000
Enter ID for employee 2: 222
Enter name for employee 2: Seeta
Enter salary for employee 2: 60000
Enter ID for employee 3: 333
Enter name for employee 3: Geeta
Enter salary for employee 3: 55000
Enter ID for employee 4: 444
Enter name for employee 4: Raj
Enter salary for employee 4: 70000
Enter ID for employee 5: 555
Enter name for employee 5: Neeta
Enter salary for employee 5: 65000
Employee with the maximum salary: Raj
ubuntu@ubuntu-OptiPlex-5000:~$
*/
//////////////////////////////SLIP NO 7///////////////////////////
// Q1. Create a Hash table containing Employee name and Salary. Display the details of the hash
// table.
// [10 Marks]

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
public class EmployeeSalaryHashTable {
public static void main(String[] args) {
// Create a HashMap to store employee names and salaries
HashMap<String, Double> employeeSalaries = new HashMap<>();
Scanner scanner = new Scanner(System.in);
// Prompt the user for the number of employees
System.out.print("Enter the number of employees: ");
int n = scanner.nextInt();
scanner.nextLine(); // Consume newline
// Input employee names and salaries
for (int i = 0; i < n; i++) {
System.out.print("Enter name for employee " + (i + 1) + ": ");
String name = scanner.nextLine();
System.out.print("Enter salary for " + name + ": ");
double salary = scanner.nextDouble();
scanner.nextLine(); // Consume newline
// Store in the hash table
employeeSalaries.put(name, salary);
}
// Display the details of the hash table
System.out.println("\nEmployee Salary Details:");
for (Map.Entry<String, Double> entry : employeeSalaries.entrySet()) {
System.out.println("Name: " + entry.getKey() + ", Salary: " + entry.getValue());
}// Close the scanner
scanner.close();
}
}

/*OUTPUT
buntu@ubuntu-OptiPlex-5000:~$ javac EmployeeSalaryHashTable.java
ubuntu@ubuntu-OptiPlex-5000:~$ java EmployeeSalaryHashTable
Enter the number of employees: 3
Enter name for employee 1: Raj
Enter salary for Raj: 50000
Enter name for employee 2: Ram
Enter salary for Ram: 60000
Enter name for employee 3: Shyam
Enter salary for Shyam: 70000
Employee Salary Details:
Name: Shyam, Salary: 70000.0
Name: Raj, Salary: 50000.0
Name: Ram, Salary: 60000.0
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Define a class student having rollno, name and percentage. Define Default and
// parameterized constructor. Accept the 5 student details and display it. (Use this keyword).
// [20 Marks]
import java.util.Scanner;
class Student {
private int rollNo;
private String name;
private double percentage;
// Default constructor
public Student() {
this.rollNo = 0;
this.name = "";
this.percentage = 0.0;
}
// Parameterized constructor
public Student(int rollNo, String name, double percentage) {
this.rollNo = rollNo;
this.name = name;
this.percentage = percentage;
}
// Method to display student details
public void displayDetails() {
System.out.println("Roll No: " + rollNo + ", Name: " + name + ", Percentage: " + percentage);
}
}
public class StudentDetails {public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
// Array to store student objects
Student[] students = new Student[5];
// Accept details for 5 students
for (int i = 0; i < 5; i++) {
System.out.println("Enter details for student " + (i + 1) + ":");
System.out.print("Roll No: ");
int rollNo = scanner.nextInt();
scanner.nextLine(); // Consume newline
System.out.print("Name: ");
String name = scanner.nextLine();
System.out.print("Percentage: ");
double percentage = scanner.nextDouble();
// Create a new Student object using the parameterized constructor
students[i] = new Student(rollNo, name, percentage);
}
// Display the details of the students
System.out.println("\nStudent Details:");
for (Student student : students) {
student.displayDetails();
}
// Close the scanner
scanner.close();
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac StudentDetails.java
ubuntu@ubuntu-OptiPlex-5000:~$ java StudentDetails
Enter details for student 1:
Roll No: 10
Name: Seeta
Percentage: 70
Enter details for student 2:
Roll No: 20
Name: Geeta
Percentage: 80.5
Enter details for student 3:
Roll No: 30
Name: Neeta
Percentage: 75
Enter details for student 4:
Roll No: 40
Name: Reeta
Percentage: 60
Enter details for student 5:
Roll No: 50
Name: Tina
Percentage: 65Student Details:
Roll No: 10, Name: Seeta, Percentage: 70.0
Roll No: 20, Name: Geeta, Percentage: 80.5
Roll No: 30, Name: Neeta, Percentage: 75.0
Roll No: 40, Name: Reeta, Percentage: 60.0
Roll No: 50, Name: Tina, Percentage: 65.0
ubuntu@ubuntu-OptiPlex-5000:~$
*/

/////////////////////////////SLIP NO 10///////////////////////////
// Q1. Write a program to accept a number from user. Check whether number is prime or not.
// [10 Marks]

import java.util.Scanner;
public class PrimeCheck {
public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
// Prompt the user for a number
System.out.print("Enter a number: ");
int number = scanner.nextInt();
// Check if the number is prime
if (isPrime(number)) {
System.out.println(number + " is a prime number.");
} else {
System.out.println(number + " is not a prime number.");
}
// Close the scanner
scanner.close();
}
// Method to check if a number is prime
public static boolean isPrime(int num) {
if (num <= 1) {return false; // 0 and 1 are not prime numbers
}
for (int i = 2; i <= Math.sqrt(num); i++) {
if (num % i == 0) {
return false; // If divisible by any number other than 1 and itself
}
}
return true; // Number is prime
}
}

/*OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac PrimeCheck.java
ubuntu@ubuntu-OptiPlex-5000:~$ java PrimeCheck
Enter a number: 5
5 is a prime number.
ubuntu@ubuntu-OptiPlex-5000:~$ javac PrimeCheck.java
ubuntu@ubuntu-OptiPlex-5000:~$ java PrimeCheck
Enter a number: 6
6 is not a prime number.
ubuntu@ubuntu-OptiPlex-5000:~$
*/
// Q2. Create a package “utility”. Define a class CapitalString under “utility” package which will
// contain a method to return String with first letter capital. Create a Person class (members – name,
// city) outside the package. Display the person name with first letter as capital by making use of
// CapitalString.
// [20 Marks]

// File: utility/CapitalString.java
package utility;
public class CapitalString {
// Method to capitalize the first letter of the given string
public String capitalizeFirstLetter(String str) {
if (str == null || str.isEmpty()) {
return str; // Return as is if the string is null or empty
}
return str.substring(0, 1).toUpperCase() + str.substring(1);
}
}
// File: Person.java
import utility.CapitalString;
public class Person {
private String name;
private String city;
// Constructor
public Person(String name, String city) {
this.name = name;
this.city = city;
}
// Method to display person’s name with capitalized first letter
public void display() {CapitalString capitalString = new CapitalString();
String capitalizedName = capitalString.capitalizeFirstLetter(name);
System.out.println("Name: " + capitalizedName + ", City: " + city);
}
public static void main(String[] args) {
// Create a Person object
Person person = new Person("john", "New York");
// Display the person’s details
person.display();
}
}
////////////////////////////SLIP NO 11///////////////////////////
// Q1. Write a program create class as MyDate with dd,mm,yy as data members. Write
// parameterized constructor. Display the date in dd-mm-yy format. (Use this keyword)
// [10 Marks]

public class MyDate {
// Data members
private int dd;
private int mm;
private int yy;
// Parameterized constructor
public MyDate(int dd, int mm, int yy) {
this.dd = dd; // ’this’ keyword refers to the current object’s dd
this.mm = mm; // ’this’ keyword refers to the current object’s mm
this.yy = yy; // ’this’ keyword refers to the current object’s yy
}
// Method to display the date in dd-mm-yy format
public void displayDate() {
System.out.printf("%02d-%02d-%02d\n", this.dd, this.mm, this.yy);
}
// Main method to test the MyDate class
public static void main(String[] args) {
MyDate date = new MyDate(18, 10, 2024);
date.displayDate(); // Output: 18-10-24
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac MyDate.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MyDate
18-10-2024
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Create an abstract class Shape with methods area & volume. Derive a class Sphere (radius).
// Calculate and display area and volume.
// [20 Marks]

// Abstract class Shape
abstract class Shape {
// Abstract methods for area and volumeabstract double area();
abstract double volume();
}
// Class Sphere that extends Shape
class Sphere extends Shape {
private double radius;
// Constructor
public Sphere(double radius) {
this.radius = radius;
}
// Implementing the area method
@Override
double area() {
return 4 * Math.PI * Math.pow(radius, 2);
}
// Implementing the volume method
@Override
double volume() {
return (4.0 / 3) * Math.PI * Math.pow(radius, 3);
}
// Method to display area and volume
public void display() {
System.out.printf("Sphere with radius %.2f:\n", radius);
System.out.printf("Area: %.2f\n", area());
System.out.printf("Volume: %.2f\n", volume());
}
}
// Main class to test the Sphere class
public class Main2 {
public static void main(String[] args) {
Sphere sphere = new Sphere(5.0); // Create a Sphere with radius 5.0
sphere.display(); // Display area and volume
}
}

/* OUTPUT
buntu@ubuntu-OptiPlex-5000:~$ javac Main2.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main2
Sphere with radius 5.00:
Area: 314.16
Volume: 523.60
ubuntu@ubuntu-OptiPlex-5000:~$
*/
///////////////////////////SLIP NO 13///////////////////////////
// Q1. Construct a Linked List having names of Fruits: Apple, Banana, Guava and Orange. Display
// the contents of the List using an Iterator.
// [10 Marks]

import java.util.LinkedList;
import java.util.Iterator;public class FruitLinkedList {
public static void main(String[] args) {
// Create a LinkedList to store fruit names
LinkedList<String> fruits = new LinkedList<>();
// Add fruit names to the LinkedList
fruits.add("Apple");
fruits.add("Banana");
fruits.add("Guava");
fruits.add("Orange");
// Display the contents of the LinkedList using an Iterator
System.out.println("Fruits in the LinkedList:");
Iterator<String> iterator = fruits.iterator();
while (iterator.hasNext()) {
System.out.println(iterator.next());
}
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac FruitLinkedList.java
ubuntu@ubuntu-OptiPlex-5000:~$ java FruitLinkedList
Fruits in the LinkedList:
Apple
Banana
Guava
Orange
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Define an interface “Operation” which has methods area(),volume(). Define a constant PI
// having value 3.142. Create a class circle (member – radius) which implements this interface.
// Calculate and display the area and volume.
// [20 Marks]
// }
// }

// Operation interface
interface Operation {
// Constant PI
double PI = 3.142;
// Abstract methods
double area();
double volume();
}
// Circle class implementing Operation
class Circle implements Operation {
private double radius;// Constructor
public Circle(double radius) {
this.radius = radius;
}
// Implementing the area method
@Override
public double area() {
return PI * radius * radius; // Area = πr²
}
// Implementing the volume method (for a cylinder)
@Override
public double volume() {
// Assuming volume for a cylinder with height = 1 for this example
return area() * 1; // Volume = Base Area * Height
}
// Method to display area and volume
public void display() {
System.out.printf("Circle with radius %.2f:\n", radius);
System.out.printf("Area: %.2f\n", area());
System.out.printf("Volume: %.2f\n", volume());
}
}
// Main class to test Circle
public class Main3 {
public static void main(String[] args) {
Circle circle = new Circle(5.0); // Create a Circle with radius 5.0
circle.display(); // Display area and volume
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac Main3.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main3
Circle with radius 5.00:
Area: 78.55
Volume: 78.55
ubuntu@ubuntu-OptiPlex-5000:~$
*/
//////////////////////////SLIP NO 15///////////////////////////
// Q1. Construct a Linked List having names of Fruits: Apple, Banana, Guava and Orange. Display
// the contents of the List in reverse order using an appropriate interface.
// [10 Marks]

import java.util.LinkedList;
import java.util.ListIterator;
public class FruitLinkedList1 {
public static void main(String[] args) {
// Create a LinkedList to store fruit names
LinkedList<String> fruits = new LinkedList<>();
// Add fruit names to the LinkedListfruits.add("Apple");
fruits.add("Banana");
fruits.add("Guava");
fruits.add("Orange");
// Display the contents of the LinkedList in reverse order
System.out.println("Fruits in reverse order:");
ListIterator<String> iterator = fruits.listIterator(fruits.size());
while (iterator.hasPrevious()) {
System.out.println(iterator.previous());
}
}
}

/*OUTPUT
Fruits in reverse order:
Orange
Guava
Banana
Apple*/

// Q2. Write a program to create a super class Employee (members – name, salary). Derive a sub- class
// as Developer (member – projectname). Create object of Developer and display the detailsof it.
// [20 Marks]

// Employee class
class Employee {
// Members of the Employee class
protected String name;
protected double salary;
// Constructor
public Employee(String name, double salary) {
this.name = name;
this.salary = salary;
}
// Method to display employee details
public void display() {
System.out.println("Name: " + name);
System.out.println("Salary: " + salary);
}
}
// Developer class that extends Employee
class Developer extends Employee {
// Member of the Developer class
private String projectName;
// Constructor
public Developer(String name, double salary, String projectName) {
super(name, salary); // Call to the superclass constructor
this.projectName = projectName;
}
// Method to display developer details@Override
public void display() {
super.display(); // Display employee details
System.out.println("Project Name: " + projectName);
}
}
// Main class to test the implementation
public class Main4 {
public static void main(String[] args) {
// Create an object of Developer
Developer developer = new Developer("Alice", 75000, "Project X");
// Display the details of the developer
developer.display();
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac Main4.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main4
Name: Alice
Salary: 75000.0
Project Name: Project X
ubuntu@ubuntu-OptiPlex-5000:~$
*/
//////////////////////////SLIP NO 16///////////////////////////
// Q1. Define a class MyNumber having one private integer data member. Write a parameterized
// constructor to initialize to a value. Write isEven() to check given number is even or not. Use
// command line argument to pass a value to the object.
// [10 Marks]

public class MyNumber1 {
// Private integer data member
private int value;
// Parameterized constructor
public MyNumber1(int value) {
this.value = value;
}
// Method to check if the number is even
public boolean isEven() {
return value % 2 == 0; // Returns true if even, false otherwise
}
// Main method to execute the program
public static void main(String[] args) {
// Check if a command-line argument is provided
if (args.length != 1) {
System.out.println("Please provide a single integer value as a command-line argument.");
return;
}
try {// Parse the command-line argument to an integer
int number = Integer.parseInt(args[0]);
// Create an instance of MyNumber
MyNumber1 myNumber = new MyNumber1(number);
// Check if the number is even and display the result
if (myNumber.isEven()) {
System.out.println(number + " is even.");
} else {
System.out.println(number + " is odd.");
}
} catch (NumberFormatException e) {
System.out.println("Invalid input. Please enter a valid integer.");
}
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac MyNumber1.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MyNumber 10
Value: 10
Is Negative: false
Is Positive: true
Is Odd: false
Is Even: true
ubuntu@ubuntu-OptiPlex-5000:~$ javac MyNumber1.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MyNumber 5
Value: 5
Is Negative: false
Is Positive: true
Is Odd: true
Is Even: false
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Write a program to create a super class Employee (members – name, salary). Derive a sub- class
// Programmer (member – proglanguage). Create object of Programmer and display the details of it.
// [20 Marks]

// Employee class
class Employee {
// Members of the Employee class
protected String name;
protected double salary;
// Constructor
public Employee(String name, double salary) {
this.name = name;
this.salary = salary;
}
// Method to display employee details
public void display() {
System.out.println("Name: " + name);
System.out.println("Salary: " + salary);
}}
// Programmer class that extends Employee
class Programmer extends Employee {
// Member of the Programmer class
private String progLanguage;
// Constructor
public Programmer(String name, double salary, String progLanguage) {
super(name, salary); // Call to the superclass constructor
this.progLanguage = progLanguage;
}
// Method to display programmer details
@Override
public void display() {
super.display(); // Display employee details
System.out.println("Programming Language: " + progLanguage);
}
}
// Main class to test the implementation
public class Main5 {
public static void main(String[] args) {
// Create an object of Programmer
Programmer programmer = new Programmer("Alice", 80000, "Java");
// Display the details of the programmer
programmer.display();
}
}
/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac Main5.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main5
Name: Alice
Salary: 80000.0
Programming Language: Java
ubuntu@ubuntu-OptiPlex-5000:~$
*/

//////////////////////////SLIP NO 17///////////////////////////
// Q1. Define a class MyNumber having one private integer data member. Write a parameterized
// constructor to initialize to a value. Write isOdd() to check given number is odd or not. Use command
// line argument to pass a value to the object.
// [10 Marks]

public class MyNumber2 {
// Private integer data member
private int value;
// Parameterized constructor
public MyNumber2(int value) {
this.value = value;
}// Method to check if the number is odd
public boolean isOdd() {
return value % 2 != 0; // Returns true if odd, false otherwise
}
// Main method to execute the program
public static void main(String[] args) {
// Check if a command-line argument is provided
if (args.length != 1) {
System.out.println("Please provide a single integer value as a command-line argument.");
return;
}
try {
// Parse the command-line argument to an integer
int number = Integer.parseInt(args[0]);
// Create an instance of MyNumber
MyNumber2 myNumber = new MyNumber2(number);
// Check if the number is odd and display the result
if (myNumber.isOdd()) {
System.out.println(number + " is odd.");
} else {
System.out.println(number + " is even.");
}
} catch (NumberFormatException e) {
System.out.println("Invalid input. Please enter a valid integer.");
}
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac MyNumber2.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MyNumber2 9
9 is odd.
ubuntu@ubuntu-OptiPlex-5000:~$ javac MyNumber2.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MyNumber2 6
6 is even.
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Define a class Student with attributes rollno and name. Define default and parameterized
// constructor. Keep the count of Objects created. Create objects using parameterized constructor and
// Display the object count after each object is created.
// [20 Marks]
class Student {
// Attributes
private int rollno;
private String name;
private static int objectCount = 0; // Static variable to keep track of object count
// Default constructor
public Student() {
objectCount++; // Increment count when a new object is created}
// Parameterized constructor
public Student(int rollno, String name) {
this.rollno = rollno;
this.name = name;
objectCount++; // Increment count when a new object is created
System.out.println("Object created. Total count: " + objectCount);
}
// Method to display student details
public void display() {
System.out.println("Roll No: " + rollno + ", Name: " + name);
}
// Static method to get the count of objects created
public static int getObjectCount() {
return objectCount;
}
}
// Main class to test the Student class
public class Main6 {
public static void main(String[] args) {
// Creating Student objects using parameterized constructor
Student student1 = new Student(101, "Alice");
Student student2 = new Student(102, "Bob");
Student student3 = new Student(103, "Charlie");
// Displaying details of each student
student1.display();
student2.display();
student3.display();
// Displaying total count of objects created
System.out.println("Total Student objects created: " + Student.getObjectCount());
}
}

/*OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac Main6.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main6
Object created. Total count: 1
Object created. Total count: 2
Object created. Total count: 3
Roll No: 101, Name: Alice
Roll No: 102, Name: Bob
Roll No: 103, Name: Charlie
Total Student objects created: 3
ubuntu@ubuntu-OptiPlex-5000:~$
*/
/////////////////////////SLIP NO 19///////////////////////////
// Q1. Write a program to accept the 'n' different numbers from user and store it in array. Display
// maximum number from an array.
// [10 Marks]

import java.util.Scanner;
public class MaxNumberFinder {
public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
// Accept the number of elements
System.out.print("Enter the number of elements: ");
int n = scanner.nextInt();
// Create an array to hold the numbers
int[] numbers = new int[n];
// Accept 'n' different numbers from the user
System.out.println("Enter " + n + " different numbers:");
for (int i = 0; i < n; i++) {
numbers[i] = scanner.nextInt();
}
// Find the maximum number
int maxNumber = findMax(numbers);
// Display the maximum number
System.out.println("The maximum number is: " + maxNumber);
scanner.close();
}
// Method to find the maximum number in an array
private static int findMax(int[] array) {
int max = array[0]; // Assume the first element is the maximum
for (int i = 1; i < array.length; i++) {
if (array[i] > max) {
max = array[i]; // Update max if a larger number is found
}
}
return max;
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac MaxNumberFinder.java
ubuntu@ubuntu-OptiPlex-5000:~$ java MaxNumberFinder
Enter the number of elements: 5
Enter 5 different numbers:
50
30
70
10
20
The maximum number is: 70
ubuntu@ubuntu-OptiPlex-5000:~$
*/
// Q2. Create an abstract class “order” having members id, description. Create a subclass
// “Purchase Order” having member as customer name. Define methods accept and display.Create 3 objects each of Purchase Order. Accept and display the details.
// [20 Marks]

import java.util.Scanner;
// Abstract class Order
abstract class Order {
protected int id;
protected String description;
// Abstract methods
public abstract void accept();
public abstract void display();
}
// Subclass PurchaseOrder
class PurchaseOrder extends Order {
private String customerName;
// Method to accept details
@Override
public void accept() {
Scanner scanner = new Scanner(System.in);
System.out.print("Enter Order ID: ");
id = scanner.nextInt();
scanner.nextLine(); // Consume newline
System.out.print("Enter Order Description: ");
description = scanner.nextLine();
System.out.print("Enter Customer Name: ");
customerName = scanner.nextLine();
}
// Method to display details
@Override
public void display() {
System.out.println("Order ID: " + id);
System.out.println("Order Description: " + description);
System.out.println("Customer Name: " + customerName);
System.out.println();
}
}
// Main class to test the implementation
public class Main7 {
public static void main(String[] args) {
PurchaseOrder[] orders = new PurchaseOrder[3]; // Array to hold 3 PurchaseOrder objects
// Accept details for 3 PurchaseOrder objects
for (int i = 0; i < 3; i++) {
System.out.println("Enter details for Purchase Order " + (i + 1) + ":");
orders[i] = new PurchaseOrder(); // Create new PurchaseOrder object
orders[i].accept(); // Accept details
}// Display details of all Purchase Orders
System.out.println("\nDetails of Purchase Orders:");
for (PurchaseOrder order : orders) {
order.display(); // Display each order's details
}
}
}

/*OUTPUT
buntu@ubuntu-OptiPlex-5000:~$ javac Main7.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main7
Enter details for Purchase Order 1:
Enter Order ID: 101
Enter Order Description: Books
Enter Customer Name: Raj
Enter details for Purchase Order 2:
Enter Order ID: 202
Enter Order Description: Laptop
Enter Customer Name: Seeta
Enter details for Purchase Order 3:
Enter Order ID: 303
Enter Order Description: Stationary
Enter Customer Name: Meeta
Details of Purchase Orders:
Order ID: 101
Order Description: Books
Customer Name: Raj
Order ID: 202
Order Description: Laptop
Customer Name: Seeta
Order ID: 303
Order Description: Stationary
Customer Name: Meeta
ubuntu@ubuntu-OptiPlex-5000:~$ ^C
ubuntu@ubuntu-OptiPlex-5000:~$
*/
i
/////////////////////////SLIP NO 20///////////////////////////
// Q1. Write a program to accept 3 numbers using command line argument. Sort and display the
// numbers.
// [10 Marks]

import java.util.Arrays;
public class SortNumbers {
public static void main(String[] args) {
// Check if exactly three arguments are provided
if (args.length != 3) {
System.out.println("Please provide exactly three numbers as command-line arguments.");
return;}
try {
// Parse the command-line arguments to integers
int[] numbers = new int[3];
for (int i = 0; i < 3; i++) {
numbers[i] = Integer.parseInt(args[i]);
}
// Sort the array
Arrays.sort(numbers);
// Display the sorted numbers
System.out.println("Sorted numbers: " + numbers[0] + ", " + numbers[1] + ", " + numbers[2]);
} catch (NumberFormatException e) {
System.out.println("Invalid input. Please enter valid integers.");
}
}
}

/* output
ubuntu@ubuntu-OptiPlex-5000:~$ javac SortNumbers.java
ubuntu@ubuntu-OptiPlex-5000:~$ java SortNumbers 8 5 9
Sorted numbers: 5, 8, 9
ubuntu@ubuntu-OptiPlex-5000:~$
*/

// Q2. Create an employee class (id,name,deptname,salary). Define a default and parameterized
// constructor. Use ‘this’ keyword to initialize instance variables. Keep a count of objects created.
// Create objects using parameterized constructor and display the object count after each object is
// created. Also display the contents of each object.
// [20 Marks]

class Employee {
// Attributes
private int id;
private String name;
private String deptName;
private double salary;
private static int objectCount = 0; // Static variable to keep track of object count
// Default constructor
public Employee() {
objectCount++; // Increment count when a new object is created
}
// Parameterized constructor
public Employee(int id, String name, String deptName, double salary) {
this.id = id;
this.name = name;
this.deptName = deptName;
this.salary = salary;
objectCount++; // Increment count when a new object is created
System.out.println("Employee created. Total count: " + objectCount);
}// Method to display employee details
public void display() {
System.out.println("ID: " + id);
System.out.println("Name: " + name);
System.out.println("Department: " + deptName);
System.out.println("Salary: " + salary);
System.out.println();
}
// Static method to get the count of objects created
public static int getObjectCount() {
return objectCount;
}
}
// Main class to test the Employee class
public class Main8{
public static void main(String[] args) {
// Creating Employee objects using parameterized constructor
Employee emp1 = new Employee(101, "Alice", "HR", 60000);
Employee emp2 = new Employee(102, "Bob", "IT", 75000);
Employee emp3 = new Employee(103, "Charlie", "Finance", 50000);
// Displaying details of each employee
System.out.println("Employee Details:");
emp1.display();
emp2.display();
emp3.display();
// Displaying total count of objects created
System.out.println("Total Employee objects created: " + Employee.getObjectCount());
}
}

/* OUTPUT
ubuntu@ubuntu-OptiPlex-5000:~$ javac Main8.java
ubuntu@ubuntu-OptiPlex-5000:~$ java Main8
Employee created. Total count: 1
Employee created. Total count: 2
Employee created. Total count: 3
Employee Details:
ID: 101
Name: Alice
Department: HR
Salary: 60000.0
ID: 102
Name: Bob
Department: IT
Salary: 75000.0
ID: 103
Name: Charlie
Department: Finance
Salary: 50000.0Total Employee objects created: 3
ubuntu@ubuntu-OptiPlex-5000:~$
*/