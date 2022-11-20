import pyttsx3
import pyautogui
import time
program = """
import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
class ItemType {
	public String name;
	public double deposit;
	public double costPerDay;
	DecimalFormat d = new DecimalFormat("0.0");
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public double getDeposit() {
		return deposit;
	}
	public void setDeposit(double deposit) {
		this.deposit = deposit;
	}
	public double getCostPerDay() {
		return costPerDay;
	}
	public void setCostPerDay(double costPerDay) {
		this.costPerDay = costPerDay;
	}
	public void display(String name,double deposit, double costPerDay) {
		System.out.println(getName());
		System.out.println(d.format(getDeposit()));
		System.out.println(d.format(getCostPerDay()));
	}
}

class Main {
public static void main(String [] args) {
	ItemType obj = new ItemType();
	Scanner sc = new Scanner(System.in);
	obj.name = sc.next();
	obj.deposit = sc.nextDouble();
	obj.costPerDay = sc.nextDouble();
	obj.setName(obj.name);
	obj.setDeposit(obj.deposit);
	obj.setCostPerDay(obj.costPerDay);
	obj.display(obj.name, obj.deposit, obj.costPerDay);
}
}

"""
if __name__=='__main__':
    time.sleep(5)
    pyautogui.typewrite(program)


