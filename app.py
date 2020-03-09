#!/usr/bin/env python
"""
@package app 

This is the entry point of the application.
"""
from lib import speaker

import argparse as arg

def main():
	p = arg.ArgumentParser(description = "Song generator machine learning models")
	# p.add_argument()
	args = p.parse_args()
	return

if __name__ == "__main__":
	main()
	exit(0)