#!/bin/bash

sed -i -s 's/from wombat/from wisdem.wombat/g' *.py
sed -i -s 's/from wombat/from wisdem.wombat/g' */*.py
sed -i -s 's/from wombat/from wisdem.wombat/g' */*/*.py
sed -i -s 's/from wombat/from wisdem.wombat/g' */*/*/*.py

sed -i -s 's/import wombat/import wisdem.wombat/g' *.py
sed -i -s 's/import wombat/import wisdem.wombat/g' */*.py
sed -i -s 's/import wombat/import wisdem.wombat/g' */*/*.py
sed -i -s 's/import wombat/import wisdem.wombat/g' */*/*/*.py

sed -i -s 's/from tests/from wisdem.test.test_wombat/g' *.py
sed -i -s 's/from tests/from wisdem.test.test_wombat/g' */*.py
sed -i -s 's/from tests/from wisdem.test.test_wombat/g' */*/*.py
sed -i -s 's/from tests/from wisdem.test.test_wombat/g' */*/*/*.py

sed -i -s 's/import tests/import wisdem.test.test_wombat/g' *.py
sed -i -s 's/import tests/import wisdem.test.test_wombat/g' */*.py
sed -i -s 's/import tests/import wisdem.test.test_wombat/g' */*/*.py
sed -i -s 's/import tests/import wisdem.test.test_wombat/g' */*/*/*.py
