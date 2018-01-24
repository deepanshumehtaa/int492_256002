all:
	mkdir -p ./out/co/kulwadee/int492/lect02
	javac -cp ./src -d ./out src/co/kulwadee/int492/lect02/SimpleLinearRegression.java
run:
	java -cp ./out co.kulwadee.int492.lect02.SimpleLinearRegression
clean:
	rm ./out/co/kulwadee/int492/lect02/SimpleLinearRegression.class
