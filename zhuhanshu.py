#import jvlei
import yuce
import classfiers
print("请选择系统！")
def main():
    while(1):
        xitong_xvanze = ['1.分类系统','2.预测系统','3.聚类系统']
        print(xitong_xvanze)
        xitong_xvanze = int(input('请选择要使用的系统(用前面数字代替即可):'))
        if xitong_xvanze == 1:
            classfiers.a()
        elif xitong_xvanze == 2:
            yuce.b()
        # elif xitong_xvanze == 3:
        #     jvlei
        b=input("请选择是否继续(y/n):")
        if b == 'y':
            continue
        else:
            break
    print("感谢使用！")
if __name__ == '__main__':
    main()
