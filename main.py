# %%
engineering_ladders_levels_markdown="""## Levels

### Technology

1.  **Adopts**: actively learns and adopts the technology and tools defined by the team
2.  **Specializes**: is the go-to person for one or more technologies and takes initiative to learn new ones
3.  **Evangelizes**: researches, creates proofs of concept and introduces new technologies to the team
4.  **Masters**: has very deep knowledge about the whole technology stack of the system
5.  **Creates**: designs and creates new technologies that are widely used either by internal or external teams

### System

1.  **Enhances**: successfully pushes new features and bug fixes to improve and extend the system
2.  **Designs**: designs and implements medium to large size features while reducing the system’s tech debt
3.  **Owns**: owns the production operation and monitoring of the system and is aware of its SLAs
4.  **Evolves**: evolves the architecture to support future requirements and defines its SLAs
5.  **Leads**: leads the technical excellence of the system and creates plans to mitigate outages

### People

1.  **Learns**: quickly learns from others and consistently steps up when it is required
2.  **Supports**: proactively supports other team members and helps them to be successful
3.  **Mentors**: mentors others to accelerate their career-growth and encourages them to participate
4.  **Coordinates**: coordinates team members providing effective feedback and moderating discussions
5.  **Manages**: manages the team members’ career, expectations, performance and level of happiness

### Process

1.  **Follows**: follows the team processes, delivering a consistent flow of features to production
2.  **Enforces**: enforces the team processes, making sure everybody understands the benefits and tradeoffs
3.  **Challenges**: challenges the team processes, looking for ways to improve them
4.  **Adjusts**: adjusts the team processes, listening to feedback and guiding the team through the changes
5.  **Defines**: defines the right processes for the team’s maturity level, balancing agility and discipline

### Influence

1.  **Subsystem**: makes an impact on one or more subsystems
2.  **Team**: makes an impact on the whole team, not just on specific parts of it
3.  **Multiple Teams**: makes an impact not only his/her team but also on other teams
4.  **Company**: makes an impact on the whole tech organization
5.  **Community**: makes an impact on the tech communityy
"""

# %%
from IPython.display import Markdown
Markdown(engineering_ladders_levels_markdown)

# %%
engineering_ladders_faqs_markdown = """FAQs
====

**What if some of the people don’t meet all the points?**

That is very normal, people are usually stronger in some areas and weaker in others. The framework should not be used as a checklist to promote people but instead as guidance to have meaningful career conversations.

**What if my organization’s career ladder is different?**

Since the framework is open source, you have the opportunity to adapt it to your organization. Feel free to use the [chart template](/charts/template.png) to define your own levels.

**When is a person ready to move to the next level?**

Companies usually expect a person to be performing at the next level _consistently for several months_ before formalizing a promotion.

**How do I collect evidence to support the discussion with my direct reports?**

Different teams collect evidence in different ways. A recommended approach is to use a combination of:

*   1:1 conversations
*   Feedback from peers and other teams
*   Self-evaluation

**Could the framework provide more specific examples of behavior to support each level?**

Specific examples of behavior require knowledge about the way that the team works, the system architecture and its technology stack. It is recommended to allow each team to define their own examples.

**Why does the framework stop at level 7?**

Levels 8 and above vary drastically from company to company. Organizations of different sizes tend to assign a diverse level of scope to positions so high in their structure.

**Do you have any additional resources about the topic?**

*   [The Manager’s Path](http://shop.oreilly.com/product/0636920056843.do): Camille Fournier does an excellent job at describing the expectations and challenges of many engineering positions. Also, she provides good advice about writing a career ladder in chapter 9.
    
*   [How to Be Good at Performance Appraisals](https://store.hbr.org/product/how-to-be-good-at-performance-appraisals-simple-effective-done-right/10295): Dick Grote explains in simple terms how to define job responsibilities and how to evaluate performance (results and behaviors).
    """

Markdown(engineering_ladders_faqs_markdown)

# %%
from enum import auto, Enum, IntEnum
import numpy as np

# %%
from enum import IntEnum, auto, EnumMeta

class DescriptiveIntEnum(IntEnum):
    def __new__(cls, description):
        obj = int.__new__(cls, len(cls.__members__) + 1)
        obj._value_ = len(cls.__members__) + 1
        obj.description = description
        return obj

class Technology(DescriptiveIntEnum):
    ADOPTS = "Actively learns and adopts the technology and tools defined by the team"
    SPECIALIZES = "Is the go-to person for one or more technologies and takes initiative to learn new ones"
    EVANGELIZES = "Researches, creates proofs of concept and introduces new technologies to the team"
    MASTERS = "Has very deep knowledge about the whole technology stack of the system"
    CREATES = "Designs and creates new technologies that are widely used either by internal or external teams"

class System(DescriptiveIntEnum):
    ENHANCES = "Successfully pushes new features and bug fixes to improve and extend the system"
    DESIGNS = "Designs and implements medium to large size features while reducing the system’s tech debt"
    OWNS = "Owns the production operation and monitoring of the system and is aware of its SLAs"
    EVOLVES = "Evolves the architecture to support future requirements and defines its SLAs"
    LEADS = "Leads the technical excellence of the system and creates plans to mitigate outages"

class People(DescriptiveIntEnum):
    LEARNS = "Quickly learns from others and consistently steps up when it is required"
    SUPPORTS = "Proactively supports other team members and helps them to be successful"
    MENTORS = "Mentors others to accelerate their career-growth and encourages them to participate"
    COORDINATES = "Coordinates team members providing effective feedback and moderating discussions"
    MANAGES = "Manages the team members’ career, expectations, performance and level of happiness"

class Process(DescriptiveIntEnum):
    FOLLOWS = "Follows the team processes, delivering a consistent flow of features to production"
    ENFORCES = "Enforces the team processes, making sure everybody understands the benefits and tradeoffs"
    CHALLENGES = "Challenges the team processes, looking for ways to improve them"
    ADJUSTS = "Adjusts the team processes, listening to feedback and guiding the team through the changes"
    DEFINES = "Defines the right processes for the team’s maturity level, balancing agility and discipline"

class Influence(DescriptiveIntEnum):
    SUBSYSTEM = "Makes an impact on one or more subsystems"
    TEAM = "Makes an impact on the whole team, not just on specific parts of it"
    MULTIPLE_TEAMS = "Makes an impact not only his/her team but also on other teams"
    COMPANY = "Makes an impact on the whole tech organization"
    COMMUNITY = "Makes an impact on the tech community"

# Example of how to access a description and numeric value
example_tech = Technology.ADOPTS
print(f"Value: {example_tech.value}, Description: {example_tech.description}")


# %%
ATTRIBUTES = [Technology, Process, System, Influence, People]

# %%
import pandas as pd
import plotly.express as px

# %%
from pydantic.dataclasses import dataclass

@dataclass
class Engineer:
    technology: Technology
    system: System
    people: People
    process: Process
    influence: Influence
    title: str = None
    level: int = None
    senior: bool = None

    def to_pandas(self):
        axes = [{
            "axis": ladder.__class__.__name__, 
            "name":ladder.name, 
            "value": ladder.value,
            "description": ladder.description,
        } for field, ladder in self.enums.items()]
        return pd.DataFrame(axes).assign(**{k:v for k,v in self.__dict__.items() if not isinstance(v, Enum)})

    @property
    def enums(self):
        return {name: value for name, value in self.__dict__.items() if isinstance(value, Enum)}

    @property
    def __dataframe__(self):
        return self.to_pandas()
    
    def __sub__(self, other) -> list[int]:
        return ((this-that) for this, that in zip(self.enums.values(), other.enums.values()))

# %%
# Define Engineer instances representing different career stages
senior_engineer = Engineer(
    technology=Technology.MASTERS,
    system=System.LEADS,
    people=People.MENTORS,
    process=Process.DEFINES,
    influence=Influence.TEAM,
    title="Senior Engineer"
)

mid_level_engineer = Engineer(
    technology=Technology.SPECIALIZES,
    system=System.DESIGNS,
    people=People.SUPPORTS,
    process=Process.ENFORCES,
    influence=Influence.SUBSYSTEM,
    title="Intermediate Engineer"
)

junior_engineer = Engineer(
    technology=Technology.ADOPTS,
    system=System.ENHANCES,
    people=People.LEARNS,
    process=Process.FOLLOWS,
    influence=Influence.COMMUNITY,
    title="Jr. Engineer"    
)

# %%
import numpy as np
def distance(engineer1: Engineer, engineer2: Engineer) -> float:
    return np.sqrt(np.power(np.fromiter(engineer1 - engineer2, dtype=int), 2).sum())

# %%
# Create Engineer objects for each developer level
developer_levels = [
    Engineer(level=1, title='D1 - Developer 1', senior=False, 
             technology=Technology.ADOPTS, system=System.ENHANCES, 
             people=People.LEARNS, process=Process.FOLLOWS, influence=Influence.SUBSYSTEM),

    Engineer(level=2, title='D2 - Developer 2', senior=False, 
             technology=Technology.ADOPTS, system=System.DESIGNS, 
             people=People.SUPPORTS, process=Process.ENFORCES, influence=Influence.SUBSYSTEM),

    Engineer(level=3, title='D3 - Developer 3', senior=False, 
             technology=Technology.SPECIALIZES, system=System.DESIGNS, 
             people=People.SUPPORTS, process=Process.CHALLENGES, influence=Influence.TEAM),

    Engineer(level=4, title='D4 - Developer 4', senior=True, 
             technology=Technology.EVANGELIZES, system=System.OWNS, 
             people=People.MENTORS, process=Process.CHALLENGES, influence=Influence.TEAM),

    Engineer(level=5, title='D5 - Developer 5', senior=True, 
             technology=Technology.MASTERS, system=System.EVOLVES, 
             people=People.MENTORS, process=Process.ADJUSTS, influence=Influence.MULTIPLE_TEAMS),

    Engineer(level=6, title='D6 - Developer 6', senior=True, 
             technology=Technology.CREATES, system=System.LEADS, 
             people=People.MENTORS, process=Process.ADJUSTS, influence=Influence.COMPANY),

    Engineer(level=7, title='D7 - Developer 7', senior=True, 
             technology=Technology.CREATES, system=System.LEADS, 
             people=People.MENTORS, process=Process.ADJUSTS, influence=Influence.COMMUNITY)
]

tech_lead_levels = [
    Engineer(level=4, title='TL4 - Tech Lead 4', senior=True, 
             technology=Technology.SPECIALIZES, system=System.OWNS, 
             people=People.COORDINATES, process=Process.ADJUSTS, influence=Influence.SUBSYSTEM),

    Engineer(level=5, title='TL5 - Tech Lead 5', senior=True, 
             technology=Technology.EVANGELIZES, system=System.EVOLVES, 
             people=People.COORDINATES, process=Process.DEFINES, influence=Influence.TEAM),

    Engineer(level=6, title='TL6 - Tech Lead 6', senior=True, 
             technology=Technology.MASTERS, system=System.LEADS, 
             people=People.COORDINATES, process=Process.DEFINES, influence=Influence.MULTIPLE_TEAMS),

    Engineer(level=7, title='TL7 - Tech Lead 7', senior=True, 
             technology=Technology.MASTERS, system=System.LEADS, 
             people=People.COORDINATES, process=Process.DEFINES, influence=Influence.COMPANY)
]

technical_program_manager_levels = [
    Engineer(level=4, title='TPM4 - Technical Program Manager 4', senior=True, 
             technology=Technology.SPECIALIZES, system=System.DESIGNS, 
             people=People.COORDINATES, process=Process.ADJUSTS, influence=Influence.MULTIPLE_TEAMS),

    Engineer(level=5, title='TPM5 - Technical Program Manager 5', senior=True, 
             technology=Technology.SPECIALIZES, system=System.DESIGNS, 
             people=People.COORDINATES, process=Process.DEFINES, influence=Influence.MULTIPLE_TEAMS),

    Engineer(level=6, title='TPM6 - Technical Program Manager 6', senior=True, 
             technology=Technology.SPECIALIZES, system=System.OWNS, 
             people=People.MANAGES, process=Process.DEFINES, influence=Influence.COMPANY),

    Engineer(level=7, title='TPM7 - Technical Program Manager 7', senior=True, 
             technology=Technology.SPECIALIZES, system=System.EVOLVES, 
             people=People.MANAGES, process=Process.DEFINES, influence=Influence.COMMUNITY)
]

engineering_manager_levels = [
    Engineer(level=5, title='EM5 - Engineering Manager 5', senior=True, 
             technology=Technology.EVANGELIZES, system=System.OWNS, 
             people=People.MANAGES, process=Process.ADJUSTS, influence=Influence.TEAM),

    Engineer(level=6, title='EM6 - Engineering Manager 6', senior=True, 
             technology=Technology.EVANGELIZES, system=System.EVOLVES, 
             people=People.MANAGES, process=Process.DEFINES, influence=Influence.TEAM),

    Engineer(level=7, title='EM7 - Engineering Manager 7', senior=True, 
             technology=Technology.EVANGELIZES, system=System.EVOLVES, 
             people=People.MANAGES, process=Process.DEFINES, influence=Influence.MULTIPLE_TEAMS)
]

# %%
engineer = junior_engineer
# engineer = mid_level_engineer
# engineer = senior_engineer

# %%
engineers = developer_levels

# %%
from more_itertools import always_iterable
import textwrap

# %%
import plotly.graph_objects as go

def generate_polar_axes_annotations() -> go.Scatterpolar:
    annotation_info = zip(
        *(
            (enum.__class__.__name__, enum.value, enum.name.replace("_"," ").title(), enum.description)
            for enum in (
                Technology.__members__
                | System.__members__
                | People.__members__
                | Process.__members__
                | Influence.__members__
            ).values()
        )
    )
    theta, r, text, description = annotation_info
    description = ['<br>'.join(textwrap.wrap(desc, width=30)) for desc in description]
    annotations = go.Scatterpolar(
        r=r,
        theta=theta,
        text=text,
        mode="text",
        hovertemplate=description,
        textposition="top center",
        #opacity=0.8,
        name="Levels"
    )
    return annotations

def style_polar_figure_axes(fig: go.Figure, titlefontsize:int=30) -> go.Figure:
    annotations = generate_polar_axes_annotations()

    fig.update_layout(
        polar=dict(
            gridshape="linear",
            radialaxis=go.layout.polar.RadialAxis(
                visible=True, range=[0, 6], dtick=1, showline=False, showticklabels=False#, labelalias=labelaliases
            ),
            angularaxis=go.layout.polar.AngularAxis(
                tickfont=go.layout.polar.angularaxis.Tickfont(size=titlefontsize),  # Change font size here
                rotation=15
            ),
        ),
        # showlegend=False
    )
    fig.add_trace(annotations)
    fig.data = fig.data[::-1] # place axis labels behind
    return fig

def plot_engineering_ladders(engineers: list[Engineer] | Engineer, size: str = None, **pxkwargs) -> go.Figure:
    engineers = list(always_iterable(engineers))
    engineers_df = pd.concat(eng.__dataframe__ for eng in engineers)
    engineers_df.description = engineers_df.description.apply(lambda x: "<br>".join(textwrap.wrap(x, 30)))
    engineers_df["title_short"] = engineers_df["title"].str.split(" ").str[0]
    hover_name=pxkwargs.pop("hover_name", "name")
    hover_data=pxkwargs.pop("hover_data", ["description"])
    color = pxkwargs.pop("color", "title_short" if size == "sm" else "title")
    
    fig = px.line_polar(
        engineers_df,
        r="value",
        theta="axis",
        hover_name=hover_name,
        hover_data=hover_data,
        line_close=True,
        color=color,
        **pxkwargs
    ).update_traces(fill='toself', opacity=max(0.5, 1/len(engineers))) # this seems to break hovering
    
    # fig
    fig = style_polar_figure_axes(fig, titlefontsize=None if size == "sm" else 30)
    if size == "sm":
        #fig.update_traces(textfont=go.scatterpolar.Textfont(size=7))
        fig.update_traces(visible="legendonly", selector=lambda trace: trace.name == "Levels")
        fig.update_layout(legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            yref="container",
            yanchor="bottom",
            y=0.9,
            title="Levels",
        ))

    return fig

# %%
fig = plot_engineering_ladders(engineering_manager_levels, template="plotly_dark", height=300, size="sm")#, text="name")
fig

# %%
fig = plot_engineering_ladders(developer_levels[2], template="plotly_dark", size="sm", text="name", height=300)
fig

# %% [markdown]
# ## Using It

# %%
me = Engineer(
    technology=Technology.MASTERS,
    system=System.EVOLVES,
    people=People.MANAGES,
    process=Process.DEFINES,
    influence=Influence.COMMUNITY,
    title="me",
    senior=True,
)

# %%
report = Engineer(
    technology=Technology.SPECIALIZES,
    system=System.OWNS,
    people=People.COORDINATES,
    process=Process.CHALLENGES,
    influence=Influence.TEAM,
    senior=False,
)

# %%
def levels(min:int=1, max:int=7) -> list[Engineer]:
    return (engineer for engineer in [*developer_levels, *technical_program_manager_levels, *tech_lead_levels, *engineering_manager_levels] if min <= engineer.level <= max)

# %%
from more_itertools import take
from itertools import islice

# %%
def levels_near(engineer: Engineer, n:int) -> list[Engineer]:
    """Returns similar levels in ladders to the engineer provided
    
    Notes
    -----
    This uses a simple euclidean distance calculation ... so ... keep that in mind
    """
    
    distances_to_levels = ((level, distance(engineer, level)) for level in levels())
    sorted_levels = (l[0] for  l in sorted(distances_to_levels, key=lambda x: x[1]))
    return take(n=n, iterable=sorted_levels)

# %%
plot_engineering_ladders([me, *levels(min=5)], template="plotly_dark", height=600).update_traces(opacity=0.2, selector=lambda trace: trace.name!="Levels")

# %% [markdown]
# ### Closest Levels on the Ladder(s)

# %%
levels_near(me, n=3)

# %% [markdown]
# ## Dash App

# %%
from dash import Output, Input, State, html, Dash, dcc, ALL, MATCH, ctx
import dash_bootstrap_components as dbc

# %%
# Function to generate table
def generate_table(dataframe):
    return dbc.Table([
        # Header
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        # Body
        html.Tbody([
            html.Tr([
                html.Td(dbc.Button(level, size="sm", id={"type":"level", "index":level, "column": col}) if (level:=dataframe.iloc[i][col]) and col not in ["Level","Senior"] else level) for col in dataframe.columns
            ]) for i in range(len(dataframe))
        ])
    ], bordered=True, size="sm")


# Create the DataFrame
data = {
    "Level": [1, 2, 3, 4, 5, 6, 7],
    "Senior": ["No", "No", "No", "Yes", "Yes", "Yes", "Yes"],
    "Developer": ["D1", "D2", "D3", "D4", "D5", "D6", "D7"],
    "Tech Lead": ["", "", "", "TL4", "TL5", "TL6", "TL7"],
    "Technical Program Manager": ["", "", "", "TPM4", "TPM5", "TPM6", "TPM7"],
    "Engineering Manager": ["", "", "", "", "EM5", "EM6", "EM7"],
    dbc.Button("+",id="add_ladder"): ["" for _ in range(7)]
}

table = generate_table(pd.DataFrame(data))

# %%
color_mode_switch =  html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="switch"),
        dbc.Switch( id="switch", value=True, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="switch"),
    ]
)

# %%
from more_itertools import one

# %%
def get_engineer(level:str, ladder:str) -> Engineer:
    n = str(level)[-1]
    return one(engineer for engineer in levels() if engineer.title == f"{level} - {ladder} {n}")

# %%
intro_blurb = """This [framework](https://www.engineeringladders.com/) allows software engineering managers to have meaningful conversations with their direct reports around the expectations of each position and how to plan for the next level in their career ladder.

Although the framework uses roles and levels that are somewhat standard in the US tech industry, every company is different. Please use the information provided as a baseline and feel free adjust it to your needs.

The framework relies heavily on radar charts to visually represent the different perspectives and expectations of a given position. Click around the levels in the [Career Ladders](#Career-Ladders) section to learn more about the expectations of levels within different ladders, and use [My Ladder](#My-Ladder) to track and compare your progress against standard levels

"""

# %%
app = Dash(title="Engineering Ladders", external_stylesheets=[dbc.themes.MATERIA, dbc.icons.FONT_AWESOME])

app.layout = html.Div([
    dbc.NavbarSimple(
        brand=app.title,
        children=[
            dbc.NavItem(dbc.NavLink("The framework", href="https://www.engineeringladders.com/")),
            color_mode_switch
        ]
    ),
    dbc.Container([
        html.P(dcc.Markdown(intro_blurb)),
        dbc.Row([
            html.H2("Career Ladders", id="Career-Ladders"),
            dbc.Col([
                table  
            ], width=6),
            dbc.Col([
                html.Div(id="viewport") 
            ], width=6)

        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H2("My Ladder", id="My-Ladder"),
                *[dbc.Row([
                    html.Div(style={"height":"fit-content"}, children=[
                    #dbc.InputGroup([
                        html.Label(ladder.__name__),
                        dcc.Slider(marks={level.value: {
                            "label": level.name.title(), 
                            "style":{
                                "font-size":10, 
                                "transform": "rotate(30deg)translate(-25%,50%)",
                                #"transform":"rotate(-75deg)",
                                #"writing-mode": "vertical-rl",
                            }
                            } for level in ladder}, min=1, max=5, id={"type":"ladder_slider", "index":ladder.__name__}, value=1, step=1)
                    ])
                    ]) for ladder in ATTRIBUTES
                ],
                dcc.Markdown(id="summary-markdown"),
                # dcc.Markdown(engineering_ladders_faqs_markdown),
            ], width=3),
            dbc.Col([dbc.Row(dbc.Col(
                dbc.Button(children=dbc.Label(className="fa fa-arrow-right fa-3x"), id="right_arrow"), className="align-middle"
                ), align="center", className="h-100")], width=1),
            dbc.Col([
                dcc.Graph(figure=None, id="figure")
            ], width=8)
        ])
    ])
])

@app.callback(
    Output("figure","figure"),
    Output("summary-markdown", "children"),
    State({'type': 'ladder_slider', 'index': ALL}, 'id'),
    Input(component_id={"type":"ladder_slider", "index": ALL}, component_property="value"),
    Input("switch","value"),
)
def update_my_ladder_plot(ids, values, light): # TODO only redraw the trace for me
    sliders = {id["index"].lower(): value for id, value in zip(ids,values)}
    me = Engineer(**sliders, title="me")
    ladders_fig = plot_engineering_ladders([me,*levels_near(me, n=3)],template="plotly_dark" if not light else "plotly")
    ladders_fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.9,
        title="Levels",
    ))

    #ladders_fig.update_traces(visible="legendonly", selector=lambda trace: trace.name not in ["Levels", "me"])
    #ladders_fig.update_traces(opacity=0.5, selector=lambda trace: trace.name!="Levels")\
    ladders_fig.update_traces(opacity=0.3, selector=lambda trace: trace.name not in ["me","Levels"])\
    
    return ladders_fig, None

@app.callback(
    Output("viewport", "children"),
    Input(component_id={"type":"level", "index": ALL,  "column": ALL}, component_property="n_clicks"),
    Input("switch", "value"),
)
def click_table(_, light):
    if (button_clicked:= ctx.triggered_id) and button_clicked != "switch":
        level, ladder = button_clicked["index"], button_clicked["column"]
        n = level[-1]
        engineer = get_engineer(level=level, ladder=ladder)
        bullets = dcc.Markdown("\n".join([f"* **{level.__class__.__name__}**: {level.name}, {level.description}" for level in engineer.enums.values()]))
        figure = plot_engineering_ladders(engineer, size="sm", color=None, text="name", template="plotly" if light else "plotly_dark")
        #figure = plot_engineering_ladders(engineer, size="sm", symbol="title_short")
        graph = dcc.Graph(figure=figure)
        view = [dbc.Row([dcc.Markdown(f"## {engineer.title}")]), dbc.Row([dbc.Col(bullets), dbc.Col(graph)])]
        return view
app.clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark"); 
       return window.dash_clientside.no_update
    }
    """,
    Output("switch", "id"),
    Input("switch", "value"),
)

if __name__ == "__main__":
    app.run(debug=True, jupyter_mode="external")
else: #gunicorn
    app = app.server

# %%


# %% [markdown]
# ### TODO:
# 
# High Level
# * Explore (view and learn about ladders and levels)
# * Introspect (define your levels)
# * Compare (show yourself against chosen levels)
# * Discover (find nearby levels)
# 
# Career Ladders html.table
# * Click levels for more information
# * Pin/compare levels to include on radar plot
# * Include a "compare all"
# * (stretch) Add new ladders
# * (stretch) sparklines in table?
# 
# My Ladder
# * Sliders to define your levels
# * -> button to populate those levels into radar plot (or should this be automagic?)
# * NN should recalc with button click, not automatically
# * display pinned/compared ladder levels
# 
# Bugs
# * Fix slider size, larger text and better spacing
# * Better way to summarize / show text for your choices -- self-only view?

# %% [markdown]
# ## Other Things
# 
# * Other ladders, e.g. https://career-ladders.dev/engineering/
# * Custom mappings, like 
# 
# ```python
# level_mapping = {
#     1: "L7",
#     2: "L8",
#     3: "L9",
#     4: "L9",
#     5: "L10",
#     6: "L11",
#     7: "DE",
# }
# ```

# %% [markdown]
# 


