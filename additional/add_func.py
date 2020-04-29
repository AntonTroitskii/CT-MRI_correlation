import numpy as np
import scipy
import math
import scipy.stats as stats

grade = (0.1, 2, 7, 15, 45)


class ConfidenceInterval:
    def __init__(self, x, y, func_ap):
        self.x = x
        self.y = y
        self.func_ap = func_ap
        self.x_mean = np.mean(self.x)
        self.sum_dif2_x = sum_diff_2_mean(self.x)
        self.n = len(self.x)
        self.y_ap = [self.func_ap(i) for i in x]
        self.std_se = std_se(self.y, self.y_ap)
        self.alpha = .05
        self.m = 1
        self.t = stats.t.ppf(1 - self.alpha, self.n - self.m - 1)

    def get_ind_app_up_list(self, n=100):
        x = np.linspace(min(self.x), max(self.x), n)
        return x, [self.get_ind_app_up_x(i) for i in x]

    def get_ind_app_low_list(self, n=100):
        x = np.linspace(min(self.x), max(self.x), n)
        return x, [self.get_ind_app_low_x(i) for i in x]

    def get_mean_app_up_list(self, n=100):
        x = np.linspace(min(self.x), max(self.x), n)
        return x, [self.get_mean_app_up_x(i) for i in x]

    def get_mean_app_low_list(self, n=100):
        x = np.linspace(min(self.x), max(self.x), n)
        return x, [self.get_mean_app_low_x(i) for i in x]

    def get_mean_app_up_x(self, x):
        return self.func_ap(x) + self.__get_conf_int_mean_x(x)

    def get_mean_app_low_x(self, x):
        res = self.func_ap(x)
        test = self.__get_conf_int_mean_x(x)
        res -= self.__get_conf_int_mean_x(x)
        return res
        # return self.func_ap(x) - self.__get_conf_int_mean_x(x)

    def get_ind_app_up_x(self, x):
        return self.func_ap(x) + self.__get_conf_int_ind_x(x)

    def get_ind_app_low_x(self, x):
        return self.func_ap(x) - self.__get_conf_int_ind_x(x)

    def __get_conf_int_ind_x(self, x):
        return self.t * self.__get_std_ind_x(x)

    def __get_conf_int_mean_x(self, x):
        return self.t * self.__get_std_mean_x(x)

    def __get_std_mean_x(self, x):
        res = self.__get_func_1(x)
        res = math.sqrt(res)
        res = self.std_se * res
        return res

    def __get_func_1(self, x):
        res = x - self.x_mean
        res = math.pow(res, 2)
        res /= self.sum_dif2_x
        res += 1 / self.n
        return res

    def __get_std_ind_x(self, x):
        res = self.__get_func_1(x)
        res = math.sqrt(res + 1)
        res = self.std_se * res
        return res


def var_se(y, y_ap, m_=1):
    """
        Фукнция возвращает дисперсию остатков регрессии по данным из 2ух массивов.
        @param m_: параметр определяющий количество независимых переменных. В случае парной регресси m_ = 1.
        @param y: исходные значения зависимой переменной
        @param y_ap: расчитанные значения зависимой переменной
    """
    n_ = len(y)
    sum_diff = sum_diff_2(y, y_ap)
    var_y = sum_diff / (n_ - m_ - 1)
    return var_y


def std_se(y, y_ap, m_=1):
    """

    @param y: измеренные значения
    @param y_ap: аппроксимированные занчения
    @param m_: параметр определяющий количество независимых переменных. В случае парной регресси m_ = 1.
    @return: оценка стандартного отклонения остатков регрессии по данным из 2ух массивов
    """
    return np.sqrt(var_se(y, y_ap, m_))


def r2_coefficient(y, y_ap):
    """
    Функция возвращает значение коэффициента детерминации
    @param y: исходные значения зависимой переменной
    @param y_ap: аппроксимированные занчения зависимой переменной
    @return: коэффициент детерминации R2
    """
    return 1.0 - (sum_diff_2(y, y_ap)) / sum_diff_2_mean(y)


def sum_diff_2(y1, y2):
    """
    Возвращает суммуквадратова разницы элементов 2ух массивов.
    @param y1: 1ый массив
    @param y2: 2ой массив
    @return: сумму квадртаов поэлементной разницы 2ух массивов.
    """
    diff_list = [np.power(y1[i] - y2[i], 2) for i in range(len(y1))]
    sum_diff = np.sum(diff_list)
    return sum_diff


def sum_diff_2_mean(x):
    """
    Возрващает сумма квадратов отклонения от среднего значений из массива x.
    @type x: Сумма квадратов отклонения от среднего значения.
    """
    x_m = np.mean(x)
    return np.sum([np.power(i - x_m, 2) for i in x])


def pearson_coefficient(y, x):
    xy_mean = np.mean([i * j for i, j in zip(x, y)])
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    var_x = np.std(x, ddof=0)
    var_y = np.std(y, ddof=0)

    return (xy_mean - x_mean * y_mean) / var_x / var_y


def mean_rel_error_ap(y, y_ap):
    """
    Возрращает среднюю относительную ошибку аппроксимации Eотн_i.
    Показывает среднее отклонение расчетных занчений y_ap от фактических y.
    @param y: исходые значения зависимой переменной.
    @param y_ap: аппроксимированные значения зависисмой переменной.
    @return: средняя относительная ошибкк аппроксимации Eотн_i.
    """
    n = len(y)
    sum_ = sum([abs((i - j) / i) for i, j in zip(y, y_ap)])
    return 100 * sum_ / n


def std_param_ab_lin_regr(x, y, y_ap, m_=1):
    """

    @param x: значения независимой переменной
    @param y: значения зависимой переменной
    @param y_ap: аппроксимированные значения зависимой переменной
    @param m_:
    @return: стандартная ошибка std_a и std_b параметров(коэффициентов) регрессии.
    """
    std_se_ = std_se(y, y_ap, m_)
    sum_x2 = np.sum([np.power(i, 2) for i in x])
    sum_dif_x2 = sum_diff_2_mean(x)
    std_a = std_se_ * np.sqrt(sum_x2 / len(x) / sum_dif_x2)
    std_b = std_se_ / np.sqrt(sum_dif_x2)
    return std_a, std_b


def std_a_f(x, y, inter, slp, m_):
    std_yx = std_yx_f(x, y, inter, slp, m_)
    sum_x2 = np.sum([np.power(i, 2) for i in x])
    sum_dif_x2 = sum_diff_2_mean(x)
    result = std_yx * np.sqrt(sum_x2 / len(x) / sum_dif_x2)
    return result


def std_b_f(x, y, inter, slp, m_=1):
    std_e_ = std_yx_f(x, y, inter, slp, m_)
    sum_dif_x2 = sum_diff_2_mean(x)
    result = std_e_ / np.sqrt(sum_dif_x2)
    return result


def std_mean_from_x(x_, x_list, std_e):
    """
    Стандартная ошика аппроксимации для среднего значения.
    @param x_: прогнозируемое знчение x.
    @param x_list: массив значенний зависимой переменной.
    @param std_e: стандартная ошибка аппроксимации.
    @return: стандартная ошика аппроксимации для среднего значения.
    """
    return std_e * math.sqrt(func_ratio_1(x_, x_list))


def std_ind_from_x(x_, x_list, std_e):
    """
    Стандартная ошика аппроксимации для индивидуального значения.
    @param x_: прогнозируемое знчение x.
    @param x_list: массив значенний зависимой переменной.
    @param std_e: стандартная ошибка аппроксимации.
    @return: стандартная ошика аппроксимации для индивидуального значения.
    """
    return std_e * math.sqrt(func_ratio_2(x_, x_list))


def func_ratio_1(x_, x_list):
    """
    Возвращает значение (1/n) + (x_ - x_mean)^2 / sum(x_i - x_mean)^2.
    n - количество элементов в массиве x.
    @param x_: прогнозируемое знчение x.
    @param x_list: массив значенний зависимой переменной.
    @return:  (1/n) + (x_ - x_mean)^2 / sum(x_i - x_mean)^2
    """
    x_mean = np.mean(x_list)
    sum_ = sum_diff_2_mean(x_list)

    return 1 / len(x_list) + (x_ - x_mean) ** 2 / sum_


def func_ratio_2(x_, x_list):
    """
    Возвращает значение 1 + (1/n) + (x_ - x_mean)^2 / sum(x_i - x_mean).
    n - количество элементов в массиве x.
    @param x_: прогнозируемое знчение x.
    @param x_list: массив значенний зависимой переменной.
    @return:  1 + (1/n) + (x_ - x_mean)^2 / sum(x_i - x_mean)^2
    """
    return 1 + func_ratio_1(x_, x_list)


def is_float(value):
    """
    Определяет является ли выражение числом целым или с плавающей точкой.
    @param value: выражение, тип которого надо определить.
    @return: True - если является, False - если не является.
    """
    return isinstance(value, float) or isinstance(value, int)


# def f_conf_mean_up

def format_sign_sum_val(val: float) -> str:
    """
    Преобразование значения с плавующей точкой (положительное или отрицательное) в текстовую переменную с необходимым форматированием.
    @param val: значение с плавующей точкой
    @return: строка с заданным форматированием
    """
    if val > 0:
        return "+ {}".format(np.abs(val))
    elif val < 0:
        return "- {}".format(np.abs(val))
    else:
        return " "


# Сумма квадратов отлокнений эксперемент знчения от расчетного по уравнению линейной регрессии
def sum_var_yx_f(x, y, inter, slp):
    var_yx_list = [(np.power(y[i] - (inter + slp * x[i]), 2)) for i in range(len(x))]
    var_xy = np.sum(var_yx_list)
    return var_xy


# Оценка дисперсии остатков регрессии Y на X по уравнению линейной регрессии
def var_yx_f(x, y, inter, slp, m_):
    var_xy = sum_var_yx_f(x, y, inter, slp)
    var_xy /= len(x) - m_ - 1
    return var_xy


# Оценка стандартного отклоения остатков регрессии Y на X по уравнению линейной регрессииу
def std_yx_f(x, y, inter, slp, m_):
    std_xy = var_yx_f(x, y, inter, slp, m_)
    return np.sqrt(std_xy)


# оценка стандартного отклонения свободного коэффициента линейной регрессии
def std_a_f(x, y, inter, slp, m_):
    std_yx = std_yx_f(x, y, inter, slp, m_)
    sum_x2 = np.sum([np.power(i, 2) for i in x])
    sum_dif_x2 = sum_diff_2_mean(x)
    result = std_yx * np.sqrt(sum_x2 / len(x) / sum_dif_x2)
    return result


# оценка стандартного отклонения коэффициента линейной регрессии
def std_b_f(x, y, inter, slp, m_=1):
    std_e_ = std_yx_f(x, y, inter, slp, m_)
    sum_dif_x2 = sum_diff_2_mean(x)
    result = std_e_ / np.sqrt(sum_dif_x2)
    return result


# grade = (0.1, 2, 7, 15)
# печать таблицы соотвтествия диапазонов перегрузки и чисел HU
def print_HU_grade(inter, slp):
    list(map(lambda x: (x - inter) / slp, grade))
    func = lambda x: (x - inter) / slp
    for i in range(0, len(grade) - 1):
        print(
            "LIC: [{} - {}] [Fe]/г -> [{} - {}] HU".format(
                grade[i], grade[i + 1], func(grade[i]), func(grade[i + 1])
            )
        )


# оценка дисперсии отдельного наблюдения (прогносзного индивидуального значения)
def var_s_i(x_i, std, mean, sum_diff2, n_):
    return std * ((1.0 / n_) + (np.power(x_i - mean, 2) / sum_diff2) + 1)


def std_s_i(x, std, mean, diff2, n_):
    return np.sqrt(var_s_i(x, std, mean, diff2, n_))


# возвращает коээфициент детерминации R^2
def get_R2(x, y, inter, slp):
    return 1.0 - (sum_var_yx_f(x, y, inter, slp)) / (sum_diff_2_mean(y))


def count_turn_points(e) -> int:
    """
    Определяет количество поворотных точек
    @return:
    @param e: массив остатокв регрессии
    @return: количество поворотных точек
    """
    count = 0
    for i in range(1, len(e) - 1):
        if (e[i] > e[i - 1]) and (e[i] > e[i + 1]):
            count += 1
        else:
            if (e[i] < e[i - 1]) and (e[i] < e[i + 1]):
                count += 1
    return count


def cout_sign_series(a):
    count_series = 1
    max_count_one_sign = 1
    count_one_sign = 1
    for index in range(1, len(a)):
        if (a[index] * a[index - 1]) < 0:
            count_series += 1
            if count_one_sign > max_count_one_sign:
                max_count_one_sign = count_one_sign
                count_one_sign = 1
        else:
            count_one_sign += 1
    return count_series, max_count_one_sign


def sortSecond(val):
    return val[1]


def get_std_e_days_list(st, fin, step, index_tab, df):
    result_list = []
    for i in range(st, fin, step):
        not_nan = (
                df[index_tab[0]].notnull()
                & df[index_tab[1]].notnull()
                & df[index_tab[2]].isna()
                & (np.abs(df[index_tab[3]]) <= i)
        )
        sub_df_regr = df[not_nan]
        sub_df_regr = sub_df_regr[index_tab[0:2]]
        [list_xt, list_yt] = [sub_df_regr[col].to_list() for col in index_tab[0:2]]
        slope_t, intercept_t, r_value_t, p_value_t, std_err_t = scipy.stats.linregress(
            list_xt, list_yt
        )
        std_e = std_yx_f(list_xt, list_yt, intercept_t, slope_t, 1)
        result_list.append([i, std_e])
    return result_list


def get_plot_lists_from_2d_arrray(array):
    list_x = [x[0] for x in array]
    list_y = [x[1] for x in array]

    return [list_x, list_y]
