#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/4/8 21:20
# @Author: Tu Xinglong
# @File  : MySQLConn.py

import pymysql


class DBCONN(object):
    '''
    对MySQL数据的增删改查操作进行封装
    '''

    def __init__(self):
        self.address = "localhost"
        self.name = "root"
        self.password = "123456"
        self.database = "market_information"

    def connectdb(self):
        print('连接到mysql服务器...')
        # 打开数据库连接
        db = pymysql.connect(self.address, self.name, self.password,
                             self.database)
        print('连接上了!')
        return db

    def createtable(self, db):
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # 如果存在表Sutdent先删除
        # cursor.execute("DROP TABLE IF EXISTS Student")
        sql = """CREATE TABLE Student (
                ID CHAR(10) NOT NULL,
                Name CHAR(8),
                Grade INT )"""
        # 创建Sutdent表
        cursor.execute(sql)

    def insertdb(self, db):
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        sql = """INSERT INTO Student
             VALUES ('001', 'CZQ', 70),
                    ('002', 'LHQ', 80),
                    ('003', 'MQ', 90),
                    ('004', 'WH', 80),
                    ('005', 'HP', 70),
                    ('006', 'YF', 66),
                    ('007', 'TEST', 100)"""

        #sql = "INSERT INTO Student(ID, Name, Grade) \
        #    VALUES ('%s', '%s', '%d')" % \
        #    ('001', 'HP', 60)
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            # Rollback in case there is any error
            print('插入数据失败!')
            db.rollback()

    def querydb(self, db):
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 查询语句
        #sql = "SELECT * FROM Student \
        #    WHERE Grade > '%d'" % (80)
        sql = "SELECT * FROM Student"
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            results = cursor.fetchall()
            for row in results:
                ID = row[0]
                Name = row[1]
                Grade = row[2]
                # 打印结果
                print("ID: %s, Name: %s, Grade: %d" % (ID, Name, Grade))
        except:
            print("Error: unable to fecth data")

    def deletedb(self, db):
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 删除语句
        sql = "DELETE FROM Student WHERE Grade = '%d'" % (100)

        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 提交修改
            db.commit()
        except:
            print('删除数据失败!')
            # 发生错误时回滚
            db.rollback()

    def updatedb(self, db):
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 更新语句
        sql = "UPDATE Student SET Grade = Grade + 3 WHERE ID = '%s'" % ('003')

        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            print('更新数据失败!')
            # 发生错误时回滚
            db.rollback()

    def closedb(self, db):
        db.close()


def main():
    dbconn = DBCONN()
    db = dbconn.connectdb()  # 连接MySQL数据库
    dbconn.createtable(db)  # 创建表
    dbconn.insertdb(db)  # 插入数据
    print('\n插入数据后:')
    dbconn.querydb(db)
    dbconn.deletedb(db)  # 删除数据
    print('\n删除数据后:')
    dbconn.querydb(db)
    dbconn.updatedb(db)  # 更新数据
    print('\n更新数据后:')
    dbconn.querydb(db)

    dbconn.closedb(db)  # 关闭数据库


if __name__ == '__main__':
    main()
